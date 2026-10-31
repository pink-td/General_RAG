"""
实文档向量的存储功能
分为以下3个模块：
初始化与集合管理：创建或加载Milvus向量数据库集合。
    初始化方法：初始化VectorStore类的实例，设置基本参数并调用集合创建或加载方法
    创建或加载集合方法：检查并创建或加载Milvus集合，定义字段结构和索引参数
文档向量化与存储：将分块后的文档转换为向量并存储。
    添加文档方法：将分块后的文档转换为向量并存储到Milvus集合
混合检索与重排序：结合稠密和稀疏向量进行检索，并通过重排序优化结果。

"""


# core/vector_store.py

import datetime

from attr.validators import max_len
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
from base.config import single_config as config
from base.logger import single_logger as logger
import sys
import os
import torch


# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    """
    需求：初始化VectorStore
    思路步骤：
    1. 构造milvus连接相关的参数: host、port、db、collection_name
    2. 构造嵌入模型
    3. 构造rerank模型
    4. 创建或加载milvus集合，保证milvus表存在
    """

    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 milvus_host=config.MILVUS_HOST
                 , milvus_port=config.MILVUS_PORT
                 , milvus_database_name=config.MILVUS_DATABASE_NAME
                 , milvus_collection_name=config.MILVUS_COLLECTION_NAME
                 , dim=1024
                 ):
        # 1. 构造milvus连接相关的参数: host、port、db、collection_name.
        # 设置 Milvus 主机地址
        self.milvus_host = milvus_host
        # 设置 Milvus 端口号
        self.milvus_port = milvus_port
        # 设置 Milvus 数据库名称
        self.milvus_database_name = milvus_database_name
        # 设置 Milvus 集合名称
        self.milvus_collection_name = milvus_collection_name
        self.dim = dim

        # 等价于MilvusClient(地址) client.use_database(数据库名)
        self.milvus_client = MilvusClient(uri=f'http://{self.milvus_host}:{self.milvus_port}', db_name=self.milvus_database_name)

        # 创建或加载milvus集合，保证milvus表存在
        self._create_or_load_collection()

        # 2. 构造嵌入模型
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        embedding_model_path = os.path.join(config.MODELS_DIR, 'bge-m3')
        # 构造bge-m3模型，通过路径加载的方式把模型加载到内存中
        self.embedding_model = BGEM3EmbeddingFunction(  # BGEM3EmbeddingFunction 是Milvus的包
            # 传入模型名: 自动下载； 传入路径:直接加载
            model_name_or_path=embedding_model_path,
            # f16: 半精度浮点数 这里取false相当于使用fp32
            # TODO 是否开启半精度
            use_fp16=False,
            # 推理（预测）设备
            # TODO 专业卡：A100, H100, A800. 消费卡：RTX4090, RTX5090
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f'加载embedding模型成功！')

        # 3. 构造rerank模型
        # bge-reranker-large模型: 用于接收query + context, 输出分数, 但是排序需要手动排序
        rerank_model_path = os.path.join(config.MODELS_DIR, 'bge-reranker-large')
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        # TODO device代表设备： mps:m1系列的mac/ cpu: cpu / cuda: nvidia的gpu。和操作系统无关
        # CrossEncoder: 交叉编码器, query 和context来做对比
        self.rerank_model = CrossEncoder(rerank_model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'加载rerank模型成功')

    """
    需求：创建或加载milvus集合，保证milvus表存在
    思路步骤：
    1. 判断milvus表是否存在
    2. 如果表不存在，创建milvus表： 
        2.1 建表： 主键、子块内容、稠密向量、稀疏向量、父块id、父块内容、资源、时间戳
        2.2 索引：稠密向量（IVF_FLAT + IP + nlist=128） 、稀疏向量（SPARSE_INVERTED_INDEX + IP + drop_radio_build ）
    3. 加载milvus集合到内存
    """
    def _create_or_load_collection(self):
        #  1. 判断milvus表是否存在
        if not self.milvus_client.has_collection(self.milvus_collection_name):
            logger.info(f'表{self.milvus_collection_name}不存在，执行建表逻辑')
            try:
                #  2. 如果表不存在，创建milvus表：
                #  2.1 建表： 主键、子块内容、稠密向量、稀疏向量、父块id、父块内容、资源、时间戳
                # 创建集合 Schema，禁用自动 ID，启用动态字段
                schema = self.milvus_client.create_schema(auto_id=False, enable_dynamic_field=True)

                # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
                # TODO id 无序，基于子块的内容生成唯一的id。  (文本1) -> md5 -> (05ff)
                schema.add_field(field_name='id', datatype=DataType.VARCHAR, max_length=100, is_primary=True)

                # 子块内容
                # 添加文本字段，VARCHAR 类型，最大长度 4096
                schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=4096)

                # 稠密向量: 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
                schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=self.dim)

                # 稀疏向量: 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
                schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)

                # 添加父块 ID 字段，VARCHAR 类型，最大长度 32
                schema.add_field(field_name='parent_id', datatype=DataType.VARCHAR, max_length=32)

                # 添加父块内容字段，VARCHAR 类型，最大长度 10240
                schema.add_field(field_name='parent_content', datatype=DataType.VARCHAR, max_length=10240)

                # 添加资源类别字段，VARCHAR 类型，最大长度 50
                # TODO 资源后续用来过滤，可以改成分区字段
                schema.add_field(field_name='subject', datatype=DataType.VARCHAR, max_length=50)

                # 添加时间戳字段，VARCHAR 类型，最大长度 50
                schema.add_field(field_name='timestamp', datatype=DataType.VARCHAR, max_length=50)

                #  2.2 索引：稠密向量（IVF_FLAT + IP + nlist=128） 、稀疏向量（SPARSE_INVERTED_INDEX + IP + drop_ratio_build ）
                index_params = self.milvus_client.prepare_index_params()

                # 稠密向量（IVF_FLAT + IP + nlist=128) 管语义
                # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
                index_params.add_index(
                    field_name='dense_vector'
                    ,index_name='dense_vector_index'
                    ,metric_type='IP'
                    ,index_type='IVF_FLAT'
                    ,params={'nlist': 128} # 128个簇
                )

                # 稠密向量: 管词频
                # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
                index_params.add_index(
                    field_name='sparse_vector'
                    ,index_name='sparse_vector_index'
                    ,metric_type='IP'
                    ,index_type='SPARSE_INVERTED_INDEX' # 稀疏向量最常用的索引类型： 倒排索引（全文检索索引）
                    ,params={
                        'drop_ratio_build': 0.2 # 忽略文档内容尾部权重的20%
                    }
                )

                # 建表:创建 Milvus 集合，应用定义的 Schema 和索引参数
                self.milvus_client.create_collection(
                    collection_name=self.milvus_collection_name,
                    schema=schema,
                    index_params=index_params
                )

                logger.info(f'执行建表：{self.milvus_collection_name}成功')
            except Exception as e:
                logger.error(f'执行建表失败：{self.milvus_collection_name}, 原因：{e}')

        #  3. 加载milvus集合到内存, 确保可立即查询
        try:
            self.milvus_client.load_collection(self.milvus_collection_name)
            logger.info(f'集合:{self.milvus_collection_name}加载完毕！')
        except Exception as e:
            logger.error(f'执行加载集合{self.milvus_collection_name}失败, 原因{e}')

    """
    需求：将子块添加到milvus中
    思路：
    1. 从子块中获取到的所有的文本
    2. 将文本输入给embedding模型，得到稀疏向量和稠密向量
    3. 构造milvus要插入的数据
        3.1 主键：子块文本进行md5
        3.2 向量：稠密向量、稀疏向量
        3.3 其他信息：子块内容、父块ID、父块内容、时间戳、资源
    4. 使用分批upsert插入milvus
    5. 支持批量处理，避免内存溢出
    """
    # 定义方法，向向量存储添加文档
    def add_documents(self, documents, batch_size=1000):
        """
        :param documents: 分块以后的子块内容 list[Document]
            Document(page_content=正文, metadata={父id、子id、父块内容parent_content、时间戳、路径、资源(source)})
        :param batch_size: 批处理大小，默认1000，可根据内存情况调整
        :return:  None 无返回值
        """
        total_docs = len(documents)
        logger.info(f"开始处理 {total_docs} 个文档，批次大小: {batch_size}")

        # 分批处理文档
        for start_idx in range(0, total_docs, batch_size):
            end_idx = min(start_idx + batch_size, total_docs)
            batch_docs = documents[start_idx:end_idx]

            logger.info(f"正在处理批次 {start_idx // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}, "
                        f"文档范围: {start_idx}-{end_idx - 1}")

            # 提取当前批次文档的内容文本列表
            texts = [child_chunk.page_content for child_chunk in batch_docs]

            # 使用 BGE-M3 嵌入函数生成文档的嵌入 ，传入list[str]
            # TODO 对于不同的嵌入模型，因为作者的设计是不一样的，没法统一。对于不同的模型，需要自己去了解对应的数据格式
            # embeddings : 字典类型。 sparse/dense
            # 假设使用text-embedding-v4（阿里） -> [n, 1024] 矩阵, 只有稠密向量

            # TODO：稀疏向量， 关键词对应的词频向量
            # embeddings['sparse'][i].indices = 第i个文档的所有的关键词 返回个一个
            # embeddings['sparse'][i].data = 第i个文档所有的关键词对应的权重

            # TODO：稠密向量， 句子转成的语义向量
            # embeddings['dense'] ->  二维矩阵 [n , 1024] n个子块对应的句子向量，一个句子 -> 1024维的向量

            try:
                embeddings = self.embedding_model(texts)

                # 初始化空列表，用于存储插入的数据
                data = []
                # 遍历当前批次的每个文档，带上索引 i
                for i, child_chunk in enumerate(batch_docs):
                    # 生成文档内容的 MD5 哈希值，作为唯一 ID
                    # hashlib.md5(): python库自带的一个通用的加密算法，能够将一串字符串文本转成一个32位，16进制的数字
                    # md5(x1) -> y1
                    # TODO 基于md5 hash算法，给子块生成一个唯一的id
                    text_hash = hashlib.md5(child_chunk.page_content.encode('utf-8')).hexdigest()

                    # 获取第 i 行的稀疏向量数据
                    sparse_row = embeddings['sparse'][i]
                    # 初始化稀疏向量字典
                    sparse_vector = {}
                    # TODO 示例，实际计算的时候原理类似，但是会有一些算法做了优化
                    # 文本1 -> (单词1=888，单词2=9，单词3=1001 ) 词表大小=2万。 -> {888:1, 9:1, 1001:1} -> 归一化 {888:0.33,9:0.33,1001:0.33}
                    # 获取稀疏向量的非零值索引
                    # col: [888,9,1001]
                    indices = sparse_row.col
                    # 获取稀疏向量的非零值
                    # data: [0.33,0.33,0.33]
                    values = sparse_row.data
                    # 将索引和值配对，填充稀疏向量字典
                    for idx, value in zip(indices, values):
                        sparse_vector[int(idx)] = float(value)

                    # 创建数据字典，包含所有字段
                    data.append({
                        # 通过hash算法生成的唯一ID
                        "id": text_hash,
                        # 子块的原文
                        "content": child_chunk.page_content,
                        # 1024维的稠密向量
                        "dense_vector": embeddings["dense"][i].tolist(),  # 转换为list以确保兼容性
                        # dict类型的稀疏向量
                        "sparse_vector": sparse_vector,
                        # 父块ID
                        "parent_id": child_chunk.metadata["parent_id"],
                        # 父块内容
                        "parent_content": child_chunk.metadata["parent_content"],
                        # 资源
                        "subject": child_chunk.metadata.get("subject", "unknown"),
                        # 时间戳
                        "timestamp": child_chunk.metadata.get("timestamp", "unknown"),
                    })
                # 检查是否有数据需要插入
                if data:
                    # 使用 upsert 操作插入数据，覆盖重复 ID
                    self.milvus_client.upsert(collection_name=self.milvus_collection_name, data=data)
                    # 记录插入或更新的文档数量日志
                    logger.info(f"批次完成: 已插入或更新 {len(data)} 个文档 (批次 {start_idx // batch_size + 1})")
                else:
                    logger.warning(f"批次 {start_idx // batch_size + 1} 中没有数据需要插入")

            except Exception as e:
                logger.error(f"处理批次 {start_idx // batch_size + 1} 时发生错误: {str(e)}")
                # 继续处理下一个批次而不是中断整个过程
                continue

        logger.info(f"完成处理 {total_docs} 个文档")

    """
    需求：实现输入query，返回排序后的topK个父块
    实现思路：
    1. 将query输入给embedding模型，得到稀疏和稠密向量
    2. 将稀疏和稠密向量构造检索对象
        2.1 稠密向量：稠密向量、度量类型（IP）、自定义参数（nprobe = 16）、 k、资源过滤条件
        2.2 稀疏向量：稀疏向量、度量类型（IP）、自定义参数（）、 k、资源过滤条件
    3. 执行混合检索，得到去重父块:  使用WeightedRanker融合结果，调用hybrid_search()
        3.1 执行混合检索得到查询结果
        3.2 对父块进行去重复
    4. 将去重后的父块再进行重排序: 使用CrossEncoder:reranker重新排序父文档
    """
    def hybrid_search_with_rerank(self,
                                  query
                                  # k 粗排
                                  , k=config.RETRIEVAL_K
                                  # m 精排
                                  , m=config.CANDIDATE_M
                                  , subject_filter=None
                                  ):
        """
               对输入的query进行混合检索
               :param query: 用户的查询字符串（原始的或者是被改写过的）
               :param k: milvus返回的limit值(检索返回数量)
               :param m: milvus返回的limit值(最终候选数量)
               :param source_filter: 资源过滤条件
               :return: 返回父块candidate_m个(上下文)
        """
        # 1. 将query输入给embedding模型，得到稀疏和稠密向量
        # TODO 使用BGE-M3生成 ① 稠密 ②稀疏向量 （文档写入milvus的也执行过同样的操作）,这是为了保证一样的文本转换成向量的数值完全一致
        embeddings = self.embedding_model([str(query)])

        # 获得稠密向量
        # 这里的0是因为只有一条数据:query需要进行embedding模型接受的格式是[batch_size]，
        # 为了满足这个格式, 把query转成了[batch_size=1]数据
        dense_vector = embeddings['dense'][0]

        # 获得稀疏向量
        sparse_vector = {}
        embeddings_sparse_ = embeddings['sparse'][[0], :]
        # col = embeddings_sparse_.col
        col = embeddings_sparse_.indices
        data = embeddings_sparse_.data
        for token_id, value in zip(col, data):
            sparse_vector[token_id] = value

        # 2. 将稀疏和稠密向量构造检索对象
        filter_expr = f"subject == {subject_filter}" if subject_filter else ""

        # 2.1 稠密向量：字段名、稠密向量、度量类型（IP）、自定义参数（nprobe = 16）、 k、资源过滤条件
        # nprobe：查询最近的几个簇
        dense_search_request = AnnSearchRequest(anns_field='dense_vector', data=[dense_vector],
                                                param={'metric_type': 'IP', 'params': {'nprobe': 10}}, limit=k, expr=filter_expr)

        # 2.2 稀疏向量：字段名、稀疏向量、度量类型（IP）、自定义参数（）、 k、资源过滤条件
        sparse_search_request = AnnSearchRequest(anns_field='sparse_vector', data=[sparse_vector],
                                                 param={'metric_type': 'IP','params': {}},
                                                 limit=k, expr=filter_expr)
        # 3. 执行混合检索，得到去重父块
        # 3.1 执行混合检索得到查询结果

        # 0.7: 稠密向量, 句子向量
        # 1.0: 稀疏向量, 稀疏向量给的大
        # TODO： ranker = WeightedRanker(0.7, 1.0): 用这种方式是因为认为稀疏向量（单词+权重）对检索结果影响更大
        # timestamp: 如果遇到了两个块内容冲突可以基于时间戳选择新的数据
        results = self.milvus_client.hybrid_search(collection_name=self.milvus_collection_name,
                                                   reqs=[dense_search_request, sparse_search_request],
                                                   ranker=WeightedRanker(0.7, 1.0),
                                                   output_fields=['content', 'parent_content', 'timestamp', 'subject',
                                                                  'parent_id'], limit=k)[0]
        # 3.2 对父块进行去重复

        # milvus的查询结果是二维的结构 [m=1,n=limit]
        # m = 传入了几个查询向量（批量查询几个向量）
        # n = limit ，每个查询返回的topK

        # 拿到检索到的所有的子块
        # _build_child_doc: 把milvus返回的结果封装成Document对象
        # hit: 命中的milvus数据，子块
        # hit {id , distance , entity : {content:   , parent_content: xxxx   }}
        child_chunks = [self._build_child_doc(hit['entity']) for hit in results]

        # TODO：通过子块，拿到所有的父块的内容，并进行去重
        unique_parent_docs = self._get_unique_parent_docs(child_chunks)

        # -----------------------------------到此为止，已经完成了粗排(query和父块进行对比)-------------------------------------

        if unique_parent_docs: # 查询到了父块
            # 如果父块只有一个，进行直接返回
            if len(unique_parent_docs) == 1:
                return unique_parent_docs

            # 4. 将去重后的父块再进行重排序
            # 4.1 构建pairs  [ [query, context1], [query, context2] , [query, context3]...] ==> 模型需要
            # 这里的parent_docs其实就是context
            # 如果父块超过一个，需要进行重排序： 基于query 和context的匹配程度做重排序
            # 构造： (query, context) (问题, 父块) 问答对
            # TODO 注意：这里的query是用户提出的原始的问题，context是查询到的相关上下文
            # rerank模型的作用就是基于rerank模型，再次计算query和context的相关性
            # pairs = [ [query, contex1], [query, contex2]  ,[query, contex3] ....]
            # pairs = [n ,2] , n = 参与rerank的父块的数量
            # TODO doc.page_content 表示父块内容
            pairs = [[query, doc.page_content] for doc in unique_parent_docs]

            # 4.2 输入给cross-encoder得到相似度分数
            # scores: 几个父块就对应几个分数
            scores = self.rerank_model.predict(pairs)

            # 4.3 对分数进行排序
            # 排序，按照分数的大小进行倒排。 分数高的排到前面
            # (1). zip: (0.3, 父块1), (0.5,父块2): 将分数和父块进行映射
            # (2). sorted -> (0.5, 父块2), (0.3,父块1) 。根据元组的第一个元素进行排序, reverse=True 倒序: 也就是根据 分数 作排序
            # (3). [doc for _, doc in sorted(zip(scores, unique_parent_docs), reverse=True)]: 取出排好序的父块内容
            sorted_parent_docs = [doc for _, doc in sorted(zip(scores, unique_parent_docs), reverse=True)]
        else: # 没有查询到父块
            sorted_parent_docs = []

        # TODO 最后保留m个父块作为最终的context
        # TODO 切片操作[:m.] -> 切片 相当于 只保存列表中下标从0到m的，[0,3)
        # 长度为 10的 list -> 长度不超过m
        # 类比 字符串的sub_string(0, m)

        # TODO m？ 考虑模型能支持的输入大小
        # 1. 考虑上下文(父块）大小：一个父块1200， 3个父块3600
        # 2. 考虑多轮对话
        # 总之：父块大小 + 多轮对话 + 提示词模板 < 模型上下文大小

        # TODO 注意：这里基于rerank的score排序以后得结果无论和问题的相关性多么小，总是取相对比较大的最大值。
        # TODO 这样会存在一个问题：有可能会查出来和问题完全不相干的。 所以可以在前面增加一个阈值判断

        # 返回m条数据
        # ranked_parent_docs: 相似度分数从大到小的10个父块
        # 通过切片的方式，截取前m个
        #  ranked_parent_docs[0:config.m] -> ranked_parent_docs数组截取下标从0到m, [0,m)
        # TODO 先倒排，再切片 = topM
        return sorted_parent_docs[:m]

    def _build_child_doc(self, hit):
        """
        把milvus返回的dict格式的子块转成Document格式
        :param hit: milvus返回的子块的各个字段，dict类型
        :return: Document类型的子块
        """
        return Document(
            page_content=hit['content'],
            metadata={
                'parent_id': hit['parent_id'],
                'parent_content': hit['parent_content'],
                'timestamp': hit['timestamp'],
                'subject': hit['subject']
            }
        )

    def _get_unique_parent_docs(self, child_docs):
        """
            从子块中提取父块，并进行去重
            :param child_docs:  从milvus中查询到的所有的子块
            :return:    去重后的父块
        """
        # 返回值
        unique_parent_docs = []

        parent_content_set = set()
        # 1.遍历所有的子块
        for doc in child_docs:
            # 2.拿到子块中的父块内容
            parent_content = doc.metadata.get('parent_content', doc.page_content)
            # 3. 对父块内容进行去重，并构建父块的文档对象
            # 3.1 父块内容存在且没有重复: 父块内容非空，且不重复
            if parent_content and parent_content not in parent_content_set:
                # 3.2 构建父块的对象
                unique_parent_docs.append(
                    Document(
                        page_content=parent_content,
                        metadata={
                            'parent_id': doc.metadata.get('parent_id'),
                            'subject': doc.metadata.get('subject'),
                            'timestamp': doc.metadata.get('timestamp')
                        }
                    )
                )

                # 放到set中，表示已经出现过
                parent_content_set.add(parent_content)

        return unique_parent_docs


if __name__ == '__main__':
    # from rag_qa.core.document_processor import process_documents

    # documents = process_documents('../data/ai_data')
    store = VectorStore()
    # store.add_documents(documents)

    results = store.hybrid_search_with_rerank('介绍一下大模型的发展史')
    print(results)