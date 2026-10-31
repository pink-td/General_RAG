"""
需求：实现文档从读取到分成子块的全流程
功能：
1. 加载文档并转成List[Document]
2. List[Document]文档切成父子块

"""
# core/document_processor.py

import os
# 文档加载器，把整个文档按照纯文本的形式加载成Document
from langchain_community.document_loaders import TextLoader
# 文档加载器，把markdown格式的数据，提取文本内容，转成Document对象
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
# 这里为什么导包可以直接使用project_document_loaders这一层，而不是进入到每个具体的文件中去导入？
# TODO 因为提前在__init__.py中已经把这个包下面所有的模块的类都统一做了导入
# TODO PDF、DOC、PPT、IMG 格式的数据都是我们自己写的
from rag_qa.project_document_loaders import OCRPDFLoader, OCRIMGLoader, OCRDOCLoader, OCRPPTLoader

# 导入分割器
# TODO 中文递归分割器（主要因为中文和英文的标点符号不一样，所以我们不使用langchain自带的RecursiveTextSplitter）
from rag_qa.project_text_spliter.project_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
# 支持markdown格式的切割（根据几级标题等）
from langchain_text_splitters import MarkdownTextSplitter

from base.logger import single_logger as logger
from base.config import single_config as config
from datetime import datetime


# 定义文档加载器
# 1. 在遍历文档的时候知道哪种后缀的文件类型使用哪个加载器进行处理 2. 通过 字典的.keys()我们也可以知道我们的系统支持哪些文件类型
# TODO 这里的逻辑： 通过python自带的工具读取文本 + orc读取图片里的内容。 这里缺少对于表格数据的处理
# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    '.doc': OCRDOCLoader,
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}


"""
需求：加载文档并转成List[Document]
思路步骤：
1. 遍历给定的目录
2. 判断目录下的文件是否是支持的
    2.1 对于支持的文件类型，使用对应的加载器进行加载
    2.2 对于不支持的文件类型，直接跳过
3. 给每个文档加入元数据：资源、路径（后缀名）、时间戳

"""

# 定义函数，从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path):
    """
        从指定文件夹加载多种类型文件并添加元数据，把处理文档返回，用于后续写入向量库
        :param directory_path: 要读取的目录（里面存放着需要被加载到向量库的所有文档）
        :return: 所有加载好并处理成Document类型(转成文本、 添加元数据 写入向量库的时候使用)的集合（列表）
    """
    # 初始化空列表，用于存储加载的文档
    documents = []
    # 获取支持的文件扩展名集合
    supported_extensions = document_loaders.keys()
    # 1. 遍历给定的目录
    # os.walk：递归遍历指定的路径
    # root: 当前所在的绝对路径(F:\www\python-data\project_rag\project\integrated_qa_system/rag_qa/core)
    # _(dirs): 当前目录下的文件夹(base)
    # files : 当前目录下的文件(LLM基础.pptx)

    # 获取source
    # directory_path -> (F:\www\python-data\project_rag\project\integrated_qa_system/rag_qa/data/ai_data
    # os.path.basename -> ai_data -> replace -> ai
    # 数据的路径要求： {资源名}_data

    # 从目录名提取资源类别（如 "ai_data" -> "ai"）
    # ai_data -> ai
    # os.path.basename(directory_path) /rag_qa/data/ai_data -> ai_data 获取路径最后一层
    # ai_data -> ai， 提取出来资源
    subject = os.path.basename(directory_path).replace("_data", "")
    print(subject)
    # os.walk： 递归遍历 directory_path 下面的所有文件， 一次循环处理一个目录
    # root： 当前的目录的路径（../integrated_qa_system/rag_qa/data/ai_data）
    #  _: 当前目录下面有哪些文件夹
    # files: 当前目录下面有哪些文件名 [LLM基础知识.pdf,人工智能.docx]
    # 遍历指定目录及其子目录
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            # 2. 判断目录下的文件是否是支持的
            # 构造文件的完整路径
            # 当前目录路径 + 文件名 = 当前目录下面所有文件的完整路径
            file_path = os.path.join(root, file_name)
            # 获得当前处理的这个文件的后缀 (.txt / .pdf):分割文件名和后缀  split/ext -> (文件名,后缀)
            extension = os.path.splitext(file_path)[1].lower()
            # 判断文件类型是否够是我们支持的
            if extension in supported_extensions:
                try:
                    # 2.1 对于支持的文件类型，使用对应的加载器进行加载
                    loader_class_name = document_loaders[extension]

                    # TODO 构建文件加载器对象
                    # OCRPDFLoader()
                    # 拼接文件的完整路径
                    if '.txt' == extension:
                        loader = loader_class_name(file_path=file_path, encoding='utf-8')
                    else:
                        loader = loader_class_name(file_path)

                    # 返回的就是加载器加载好的完整的文档
                    # list[Document[metadata(dict结构), page_content:大字符串 ] ]
                    # 里面只有一个元素， 就相当于 一个pdf/doc/ppt文件 -> Document对象
                    # 调用加载器加载文档内容，返回文档列表:  每个文件返回的就是一个List[Document]
                    loaded_documents = loader.load()

                    # 3. 遍历加载的每个文档: 给每个文档加入元数据：资源、路径（后缀名）、时间戳
                    # 添加一些新的元数据进去， 比如资源、文件路径、后缀名、时间。用于插入向量库使用
                    # 这里面实际就执行了一次，因为list里面只有一个元素
                    for loaded_document in loaded_documents:
                        # Document: page_content:str, 文档的原文; metadata(叫元数据，但是并不必纠结。就是给用户用来存放额外信息):dict, 文档的附加信息
                        loaded_document.metadata['timestamp'] = datetime.now().isoformat()
                        loaded_document.metadata['subject'] = subject
                        loaded_document.metadata['file_path'] = file_path
                        loaded_document.metadata['extension'] = extension

                    # 把每个文件加载的文档全部放到documents中返回
                    # extend方法：  loaded_docs: list[Document]
                    # documents.append : [ [doc1, doc2] ,[doc3, doc4]]
                    # documents.extend :  [ doc1, doc2 ,doc3, doc4]. 把loaded_docs拆开，所有元素放入documents中
                    documents.extend(loaded_documents)
                    # 记录成功加载文件的日志
                    logger.info(f"成功加载文件: {file_path}")
                except Exception as e:
                    logger.error(f'处理文件:{file_path}失败，原因:{e}')
            else:
                # 2.2 对于不支持的文件类型，直接跳过
                logger.warning(f'处理文件:{file_path}失败，不支持的类型')

    return documents


"""
需求：将文档切成父子块,处理文档并进行分层切分，返回子块结果
步骤：
1. 调用加载文档的API，获取到所有的文档
2. 构造文档的spliter(初始化文档分割器)，
    2.1 对于markdown格式，使用MarkdownTextSplitter
    2.2 对于其他格式，使用ChineseRecursiveTextSplitter
    有4种情况：md格式-父块、md格式-子块、txt格式-父块、txt格式子块
    
3. 把文件分割成多个子块, 按层次进行遍历和加载: 
    3.1 遍历所有的文档，分成父块
    3.2 遍历所有的父块，分成子块；给父块增加元数据
    3.3 遍历所有的子块，给子块增加元数据
    包括： ①parent_id ②id ③parent_content 

最终milvus表里的字段：
0. 主键          【子块提供-page_content】
1. 稠密向量（语义向量）-通过bge-m3模型生成, 固定长度列表 [0.006965979002416134,-0.019829535856842995,....]    【子块提供-page_content】
2. 稀疏向量（词频向量）-通过bge-m3模型生成, {"6":0.038418810814619064,"12":0.0549} 6:词表下标    【子块提供-page_content】
3. 父块内容     【子块提供-metadata】
4. 子块id       【子块提供-metadata】
5. 父块id       【子块提供-metadata】
6. 时间戳       【子块提供-metadata】
7. 资源         【子块提供-metadata】

"""

# 定义函数，处理文档并进行分层切分，返回子块结果
def process_documents(
        directory_path: str,
        parent_chunk_size=config.PARENT_CHUNK_SIZE,
        child_chunk_size=config.CHILD_CHUNK_SIZE,
        overlap_size=config.CHUNK_OVERLAP
):
    """
        处理文档并进行分层切分，返回子块结果
        :param directory_path:      要处理的文档路径
        :param parent_chunk_size:   父块大小
        :param child_chunk_size:    子块大小
        :param chunk_overlap:       重叠字符数
        :return: 所有的子块
    """

    # 1. 调用加载文档的API，获取到所有的文档
    # 输入: 路径
    # 输出: List[Document]
    documents = load_documents_from_directory(directory_path)

    # 2. 构造文档的spliter，有4种情况：md格式-父块、md格式-子块、txt格式-父块、txt格式子块
    # 初始化 Markdown 专用分词器
    parent_markdown_spliter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=overlap_size)
    child_markdown_spliter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=overlap_size)
    # 初始化父块和子块分词器（通用）
    parent_text_spiter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=overlap_size)
    child_text_spiter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=overlap_size)

    # 初始化空列表，用于存储所有子块
    child_chunks = []

    # 3. 按层次进行遍历和加载:
    # enumerate: 在遍历一个list等可迭代元素同时，添加当前遍历的是第几个元素
    for i, document in enumerate(documents):
        # 3.1 遍历所有的文档，分成父块
        # 获取文件扩展名
        extension = document.metadata['extension']

        # 选择切分器: 除了.md外,都是纯文本
        if_markdown = (extension == '.md')
        # 如果是.md,则使用.md分割器, 如果不是.md,则使用 纯文本分割器
        parent_spliter_to_use = parent_markdown_spliter if if_markdown else parent_text_spiter # 父块分割器
        child_spliter_to_use = child_markdown_spliter if if_markdown else child_text_spiter # 子块分割器
        logger.info(
            f"处理文档: {document.metadata['file_path']}, 使用切分器: {'Markdown' if if_markdown else 'ChineseRecursive'}")

        # 切割器需要传入的是list格式，需要转一个转换 Document -> [Document]

        # TODO parent_docs -> list[Document] 每个Document对应的一个父块
        parent_docs = parent_spliter_to_use.split_documents([document])

        # 3.2 遍历所有的父块，分成子块；给父块增加元数据
        for j , parent_doc in enumerate(parent_docs):
            # 为父块生成唯一 ID，格式为 "doc_i_parent_j"
            parent_id = f"doc_{i}_parent_{j}"
            # 将父块 ID 添加到元数据
            parent_doc.metadata["parent_id"] = parent_id
            # 将父块内容存储到元数据
            parent_doc.metadata["parent_content"] = parent_doc.page_content

            # 使用子块分词器将父块切分为子块
            current_child_chunks = child_spliter_to_use.split_documents([parent_doc])
            # 3.3 遍历所有的子块，给子块增加元数据,带上索引 k, k表示: 第i个文档的第j个父块的第k个子块
            for k , child_chunk in enumerate(current_child_chunks):
                """
                最终milvus表里的字段：
                0. 主键          【子块提供-page_content】
                1. 稠密向量（语义向量）, 固定长度列表 [0.006965979002416134,-0.019829535856842995,....]    【子块提供-page_content】
                2. 稀疏向量（词频向量）, {"6":0.038418810814619064,"12":0.0549} 6:词表下标    【子块提供-page_content】
                3. 父块内容     【子块提供-metadata】
                4. 子块id       【子块提供-metadata】
                5. 父块id       【子块提供-metadata】
                6. 时间戳       【子块提供-metadata】
                7. 资源         【子块提供-metadata】

                """
                # 为子块生成唯一 ID，格式为 "parent_id_child_k", 表示: 第i个文档的第j个父块的第k个子块
                child_id = f"{parent_id}_child_{k}"
                child_chunk.metadata['child_id'] = child_id

                # 为子块添加父块 ID 到元数据
                child_chunk.metadata['parent_id'] = parent_id # 第i个文档的第j个父块

                # 为子块添加父块内容到元数据: 把父块的内容添加到子块的元数据，是为了在向量库检索的时候直接通过子块的匹配拿到父块的全部文本内容
                child_chunk.metadata['parent_content'] = parent_doc.page_content
                child_chunk.metadata['process_time'] = document.metadata['timestamp']
                child_chunk.metadata['subject'] = document.metadata['subject']

            # 将子块添加到子块列表中
            child_chunks.extend(current_child_chunks)
    # 记录子块总数日志
    logger.info(f"子块数量: {len(child_chunks)}")

    # 返回所有子块列表
    return child_chunks


if __name__ == '__main__':
    # documents = load_documents_from_directory('../data/ai_data')
    """
    [Document(
        metadata={
            'source': '../data\\ai_data\\LLM基础知识.pdf',
             'timestamp': '2026-07-10T13:30:21.567911', 
             'subject': 'ai', 
             'file_path': '../data\\ai_data\\LLM基础知识.pdf',
            'extension': '.pdf'
            }, 
        page_content='LLM背景知识介绍\n \n学习⽬标\n \n了解LLM背景的知识\n掌...'
        ), 
     Document(
        metadata={
            'source': '../data\\ai_data\\人工智能.docx', 
            'timestamp': '2026-07-10T13:30:22.481727',
             'subject': 'ai', 
             'file_path': '../data\\ai_data\\人工智能.docx', 
             'extension': '.docx'
            }, 
        page_content='人工智能,...资源说明：\n1．补充标题3...'
        )
    )]
    2
    """
    # print(documents)
    # print(len(documents))
    documents = process_documents('../data/ai_data')

    print(documents)
    print(len(documents))