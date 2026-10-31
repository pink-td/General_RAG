from rank_bm25 import BM25Okapi # 导入 BM25 算法
from base.logger import single_logger as logger # 导入日志

from mysql_qa.utils.preprocess import preprocess_text # 导入文本预处理

from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MysqlClient

import numpy as np #  实现归一化


class Bm25Search:
    """
    需求：实现FAQ模块的数据加载部分。（从mysql到bm25）
    思路步骤：
        0. 准备bm25search需要的对象：redis_client、mysql_client、bm25search对象
        1. 判断系统是不是第一次启动
        2. 如果是第一次启动
        2.1 从mysql中拉取所有的问题
        2.2 把所有的问题写入redis中（origin_questions）
        2.3 把所有的问题调用预处理模块进行分词，写入redis中（questions）
        3. 构建bm25检索器，传入（questions）
    """

    def __init__(self, redis_client: RedisClient, mysql_client: MysqlClient):
        self.redis_client = redis_client  # 初始化 Redis 客户端
        self.mysql_client = mysql_client # 初始化 MySQL 客户端

        self.original_questions = None  # 初始化原始问题
        self.tokenized_questions = None # 初始化问题列表
        self.bm25 = None # 初始化 BM25 模型
        logger.info('系统初始化完毕，开始执行数据加载')
        self._load_data() # 加载数据

    def _load_data(self):
        # 加载数据
        original_questions_key = 'projectrag:original_questions'
        tokenized_questions_key = 'projectrag:tokenized_questions'
        #  1. 判断系统是不是第一次启动(从 Redis 获取原始问题)
        tokenized_questions = self.redis_client.get_data(tokenized_questions_key)
        original_questions = self.redis_client.get_data(original_questions_key)

        # 如果 Redis 中没有数据，从 MySQL 加载
        if not tokenized_questions or not original_questions:
            #  2. 如果是第一次启动
            logger.info('首次启动，从mysql中获取数据')
            #  2.1 从mysql中拉取所有的问题
            original_questions = self.mysql_client.fetch_all_questions()
            if not original_questions:
                # 如果mysql里面差不到数据，直接终止程序
                logger.error("mysql里找不到原始的问题数据")
                return

            #  2.2 把所有的问题写入redis中（origin_questions）(存储原始问题到 Redis)
            self.redis_client.set_data(original_questions_key, original_questions)
            #  2.3 把所有的问题调用预处理模块进行分词，写入redis中（questions） --分词问题
            tokenized_questions = [preprocess_text(question) for question in original_questions]
            # 存储分词问题到 Redis
            self.redis_client.set_data(tokenized_questions_key, tokenized_questions)
            logger.info(f'在mysql中获取数据成功，高频问题数:{len(tokenized_questions)}')

        self.original_questions = original_questions
        # 设置问题列表
        self.tokenized_questions = tokenized_questions

        #  3. 构建bm25检索器，传入（questions）
        self.bm25 = BM25Okapi(self.tokenized_questions)

    """
    需求：实现FAQ模块的查询功能，用户输入一个query，返回超过阈值的概率的问题对应的答案
    思路步骤：
    1. 判断用户的query是否合法，非空字符串
    2. 尝试去redis中查找，一模一样的问题
    3. 对query进行预处理，得到分词数据
    4. 通过bm25检索器得到query和每个问题的相似度分数：[n] 
    5. 对相似度分数进行归一化
    6. 取相似度分数归一化以后的最大值，判断是否大于给定的阈值；如果小于阈值，返回空答案
    7. 如果大于阈值，通过argmax找到最大分数对应的索引
    8. 根据索引找到对应的原始问题（python内存）
    9. 查看redis缓存中是否有该问题的答案
    10. 查看mysql中是否有该问题答案
    11. 返回答案，同时写入redis缓存
    
    简化版思路步骤：
    1. 用户的query输入FAQ模块以后，先判断是否有异常，如果有异常直接返回
    2. 将用户的query进行预处理，转小写和分词
    3. 将分词以后的query输入BM25模型，得到和每个高频问题的相似度
    4. 对相似度进行归一化，并判断最高分的文档是否大于阈值
    5. 如果大于阈值，先在redis中查询，如果查询不到，查询mysql
    6. 如果小于阈值，进行RAG模块
    """

    def search(self, query, threshold=0.85):
        """
        实现FAQ模块的查询功能，用户输入一个query，返回超过阈值的概率的问题对应的答案
        :param query:   用户的问题
        :return:        (答案, 是否要进入RAG模块)
        """

        # 1. 判断用户的query是否合法，非空字符串
        if query is None or type(query) is not str:
            # TODO 1.query不合法，不继续进入RAG
            logger.error(f'query不合法:{query}')
            return None, False
        # 2. 尝试去redis中查找，一模一样的问题
        answer = self.redis_client.get_answer(query)
        if answer:
            # TODO 2.在redis查到了一模一样的问题，直接返回
            logger.debug(f'在redis查到了一模一样的问题,问题:{query},答案:{answer}')
            return answer, False
        try:
            # 3. 对query进行预处理，得到分词数据
            tokenized_query = preprocess_text(query)
            # 4. 通过bm25检索器得到query和每个问题的相似度分数：[467]
            scores = self.bm25.get_scores(tokenized_query)
            # 5. 对相似度分数进行归一化
            scores = self._softmax(scores)
            # 6. 取相似度分数归一化以后的最大值，判断是否大于给定的阈值；如果小于阈值，返回空答案
            max_index = np.argmax(scores)
            max_score = scores[max_index]
            logger.info(f'匹配成功，分数:{max_score}')
            # 7. 如果大于阈值，通过argmax找到最大分数对应的索引
            if max_score >= threshold:
                # 8. 根据索引找到对应的原始问题（python内存）
                origin_question = self.original_questions[max_index]
                # 9. 查看redis缓存中是否有该问题的答案
                answer = self.redis_client.get_answer(origin_question)
                if answer:
                    # TODO 3. 找到了相似的问题，且大于阈值。redis的缓存中找到了对应的答案
                    return answer, False
                # 10. 查看mysql中是否有该问题答案
                answer = self.mysql_client.fetch_answer(origin_question)
                if answer:
                    # 11. 返回答案，同时写入redis缓存
                    # TODO 4. 找到了相似的问题，且大于阈值。redis的缓存没有找到答案，但是在mysql中找到了
                    self.redis_client.set_answer(origin_question, answer)
                    return answer, False
                # TODO 5. 找到了相似的问题，且大于阈值。redis和mysql都没有找到
                logger.error(f'问题已经匹配成功，但是在mysql中未找到答案:{query}')
                return None, True

            # TODO 6.没有匹配到相似度大于阈值的问题
            return None, True
        except Exception as e:
            # 记录搜索失败
            logger.error(f"搜索失败: {e}")
            # 返回 None 和 True
            return None, True

    def _softmax(self, scores):
        # scores ： 值域 [0, +无穷]
        # scores - np.max(scores): (-无穷, 0)
        # np.exp: [0,1]
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / np.sum(exp_scores)


if __name__ == '__main__':
    mysql_client = MysqlClient()
    redis_client = RedisClient()

    search = Bm25Search(
        redis_client=redis_client,
        mysql_client=mysql_client
    )
