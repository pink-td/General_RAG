# core/rag_system.py

"""
需求：实现rag完成流程

需要实现的方法：
1. 初始化
2. 3个私有方法
3. 私有方法和策略选择器的合并
4. 完成流程

"""

import time

from base.config import single_config as config
from base.logger import single_logger as logger

from rag_qa.core.vector_store import VectorStore
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.query_classifier import QueryClassifier
from rag_qa.core.strategy_selector import StrategySelector

# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate
# 导入 LangChain 消息类型
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

"""
需求：实现rag系统的初始化
思路步骤：
1. 引入vectorstore
2. 引入prompts模块
3. 引入意图识别分类器
4. 引入策略选择器
5. 初始化大模型client
"""


class RagSystem:
    def __init__(self):
        try:
            start_time = time.time()
            logger.info('正在初始化RAG系统...')
            self.prompts = RAGPrompts()
            self.query_classifier = QueryClassifier()
            self.strategy_selector = StrategySelector()
            self.vector_store = VectorStore()
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL,
                api_key=config.DASHSCOPE_API_KEY,
                base_url=config.DASHSCOPE_BASE_URL,
                temperature=0.1,
            )
            elapsed = time.time() - start_time
            logger.info(f'RAG系统初始化完成, 耗时: {elapsed:.2f}s')
        except Exception as e:
            logger.error(f'RAG系统初始化失败: {e}')
            raise

    def _call_model(self, prompt, count_down=3):
        start_time = time.time()
        logger.info(f'开始调用大模型，最大重试次数: {count_down}')
        for i in range(count_down):
            try:
                messages = [
                    SystemMessage('你是一个有用的助手，能够准确无误的完成用户的命令'),
                    HumanMessage(prompt),
                ]
                result = self.llm.invoke(messages).content
                elapsed = time.time() - start_time
                logger.info(f'大模型调用成功，第 {i + 1} 次尝试, 耗时: {elapsed:.2f}s')
                return result
            except Exception as e:
                logger.error(f'大模型调用失败，第 {i + 1}/{count_down} 次尝试，原因: {e}')
                if i < count_down - 1:
                    logger.info(f'等待 {1} 秒后重试...')
                    time.sleep(1)
                    continue
        elapsed = time.time() - start_time
        logger.warning(f'大模型调用失败，已重试 {count_down} 次，放弃调用, 总耗时: {elapsed:.2f}s')
        return None

    """
    需求: 实现假设策略的具体逻辑
    思路步骤:
    1. 将原始的query和假设问题生成的提示词模板进行结合，生成完整提示词
    2. 将提示词送给大模型，返回改写的query 
    3. 将改写的query调用混合查询接口，返回父块
    """

    def _retrieve_with_hyde(self, query, subject_filter):
        try:
            start_time = time.time()
            logger.info(f'[假设问题检索] 开始处理, query: {query}, subject_filter: {subject_filter}')
            #  1. 将原始的query和假设问题生成的提示词模板进行结合，生成完整提示词
            hyde_prompt = self.prompts.hyde_prompt()
            prompt = hyde_prompt.format(query=query)
            #  2. 将提示词送给大模型，返回改写的query
            hyde_answer = self._call_model(prompt)
            if hyde_answer:
                #  3. 将改写的query调用混合查询接口，返回父块
                logger.info(f'[假设问题检索] query改写成功，修改前:{query}, 修改后:{hyde_answer}')
                chunks = self.vector_store.hybrid_search_with_rerank(hyde_answer, subject_filter=subject_filter)
            else:
                logger.warning(f'[假设问题检索] query改写失败，使用兜底逻辑, 原始query: {query}')
                chunks = self.vector_store.hybrid_search_with_rerank(query, subject_filter=subject_filter)
            elapsed = time.time() - start_time
            logger.info(f'[假设问题检索] 检索完成，返回 {len(chunks)} 个文档块, 耗时: {elapsed:.2f}s')
            return chunks
        except Exception as e:
            logger.error(f'[假设问题检索] 处理异常: {e}', exc_info=True)
            logger.warning(f'[假设问题检索] 异常兜底，使用原始query进行检索')
            try:
                return self.vector_store.hybrid_search_with_rerank(query, subject_filter=subject_filter)
            except Exception as fallback_error:
                logger.error(f'[假设问题检索] 兜底检索也失败: {fallback_error}')
                return []

    """
    需求: 实现回溯策略的具体逻辑
    思路步骤:
    1. 将原始的query和假设问题生成的提示词模板进行结合，生成完整提示词
    2. 将提示词送给大模型，返回改写的query 
    3. 将改写的query调用混合查询接口，返回父块
    """

    def _retrieve_with_backtracking(self, query, subject_filter):
        try:
            start_time = time.time()
            logger.info(f'[回溯问题检索] 开始处理, query: {query}, subject_filter: {subject_filter}')
            #  1. 将原始的query和假设问题生成的提示词模板进行结合，生成完整提示词
            backtracking_prompt = self.prompts.backtracking_prompt()
            prompt = backtracking_prompt.format(query=query)
            #  2. 将提示词送给大模型，返回改写的query
            simple_query = self._call_model(prompt)
            if simple_query:
                #  3. 将改写的query调用混合查询接口，返回父块
                logger.info(f'[回溯问题检索] query改写成功，修改前:{query}, 修改后:{simple_query}')
                chunks = self.vector_store.hybrid_search_with_rerank(simple_query, subject_filter=subject_filter)
            else:
                logger.warning(f'[回溯问题检索] query改写失败，使用兜底逻辑, 原始query: {query}')
                chunks = self.vector_store.hybrid_search_with_rerank(query, subject_filter=subject_filter)
            elapsed = time.time() - start_time
            logger.info(f'[回溯问题检索] 检索完成，返回 {len(chunks)} 个文档块, 耗时: {elapsed:.2f}s')
            return chunks
        except Exception as e:
            logger.error(f'[回溯问题检索] 处理异常: {e}', exc_info=True)
            logger.warning(f'[回溯问题检索] 异常兜底，使用原始query进行检索')
            try:
                return self.vector_store.hybrid_search_with_rerank(query, subject_filter=subject_filter)
            except Exception as fallback_error:
                logger.error(f'[回溯问题检索] 兜底检索也失败: {fallback_error}')
                return []

    """
    需求: 实现子查询策略的具体逻辑
    思路步骤:
    1. 将原始的query和假设问题生成的提示词模板进行结合，生成完整提示词
    2. 将提示词送给大模型，返回改写的query 
    3. 将改写的query调用混合查询接口，返回父块
    """

    def _retrieve_with_subquery(self, query, subject_filter):
        try:
            start_time = time.time()
            logger.info(f'[子查询检索] 开始处理, query: {query}, subject_filter: {subject_filter}')
            # 1. 将原始的query和假设问题生成的提示词模板进行结合，生成完整提示词
            subquery_prompt = self.prompts.subquery_prompt()
            subquery_prompt = subquery_prompt.format(query=query)
            # 2. 将提示词送给大模型，返回改写的query
            subqueries = self._call_model(subquery_prompt)

            if subqueries:
                subquery_list = [subquery.strip() for subquery in subqueries.strip().split('\n') if subquery.strip()]
                logger.info(f'[子查询检索] 改写成功，生成 {len(subquery_list)} 个子查询: {subquery_list}')

                if subquery_list:
                    total_chunks = []
                    for idx, subquery in enumerate(subquery_list):
                        logger.info(f'[子查询检索] 正在处理第 {idx + 1}/{len(subquery_list)} 个子查询: {subquery}')
                        try:
                            sub_start = time.time()
                            subquery_chunks = self.vector_store.hybrid_search_with_rerank(
                                subquery, subject_filter=subject_filter
                            )
                            sub_elapsed = time.time() - sub_start
                            logger.info(f'[子查询检索] 子查询 {idx + 1} 返回 {len(subquery_chunks)} 个文档块, 耗时: {sub_elapsed:.2f}s')
                            total_chunks.extend(subquery_chunks)
                        except Exception as sub_err:
                            logger.error(f'[子查询检索] 子查询 {idx + 1} 检索失败: {sub_err}')
                            continue

                    # 根据dict key的去重的机制，最后chunk_dict保留下来的key都是唯一的
                    chunk_dict = {doc.page_content: doc for doc in total_chunks}
                    result_chunks = list(chunk_dict.values())
                    elapsed = time.time() - start_time
                    logger.info(f'[子查询检索] 检索完成，去重后返回 {len(result_chunks)} 个文档块, 总耗时: {elapsed:.2f}s')
                    return result_chunks
                else:
                    logger.warning(f'[子查询检索] 子查询拆分结果为空，使用兜底逻辑')
                    return self.vector_store.hybrid_search_with_rerank(query, subject_filter=subject_filter)
            else:
                logger.warning(f'[子查询检索] query改写失败，使用兜底逻辑, 原始query: {query}')
                return self.vector_store.hybrid_search_with_rerank(query, subject_filter=subject_filter)
        except Exception as e:
            logger.error(f'[子查询检索] 处理异常: {e}', exc_info=True)
            logger.warning(f'[子查询检索] 异常兜底，使用原始query进行检索')
            try:
                return self.vector_store.hybrid_search_with_rerank(query, subject_filter=subject_filter)
            except Exception as fallback_error:
                logger.error(f'[子查询检索] 兜底检索也失败: {fallback_error}')
                return []

    """
    需求：实现策略分类和具体策略逻辑的整合
    思路步骤：
    1. 先对策略进行分类
    2. 针对不同的策略进入不同的分支
    3. 对处理结果进行保护性的截断
    """

    def retrieve_and_merge(self, query, subject_filter=None, strategy=None):
        try:
            start_time = time.time()
            logger.info(f'[策略合并] 开始处理, query: {query}, subject_filter: {subject_filter}, strategy: {strategy}')

            # 1. 先对策略进行分类
            if strategy is None:
                strategy = self.strategy_selector.select_strategy(query)
            logger.info(f'[策略合并] 当前使用策略: {strategy}')

            # 2. 针对不同的策略进入不同的分支
            chunks = []
            if strategy == '假设问题检索':
                chunks = self._retrieve_with_hyde(query, subject_filter=subject_filter)
            elif strategy == '回溯问题检索':
                chunks = self._retrieve_with_backtracking(query, subject_filter=subject_filter)
            elif strategy == '子查询检索':
                chunks = self._retrieve_with_subquery(query, subject_filter=subject_filter)
            else:
                logger.info(f'[策略合并] 未匹配策略，使用默认混合检索')
                chunks = self.vector_store.hybrid_search_with_rerank(query, subject_filter=subject_filter)

            # 3. 对处理结果进行保护性的截断
            # 对于子查询特殊情况的截断： 给一个固定的数 n * M ，子查询数 * M， 不做限制
            result_chunks = chunks[:config.CANDIDATE_M] if strategy != '子查询检索' else chunks[: 5 * config.CANDIDATE_M]

            elapsed = time.time() - start_time
            logger.info(f'[策略合并] 检索完成，截断前 {len(chunks)} 个文档块，截断后 {len(result_chunks)} 个文档块, 耗时: {elapsed:.2f}s')
            return result_chunks
        except Exception as e:
            logger.error(f'[策略合并] 处理异常: {e}', exc_info=True)
            logger.warning(f'[策略合并] 异常兜底，使用原始query进行默认检索')
            try:
                return self.vector_store.hybrid_search_with_rerank(query, subject_filter=subject_filter)[:config.CANDIDATE_M]
            except Exception as fallback_error:
                logger.error(f'[策略合并] 兜底检索也失败: {fallback_error}')
                return []

    """
    需求：实现从query输入到答案输出的完整流程
    思路：
    1. 对query进行二分类
    2. 对于通用问题，直接调用模型回答
    3. 对于专业问题，调用“策略分类和具体策略逻辑的整合”返回
    
    """

    def generate_answer(self, query, subject_filter=None):
        try:
            start_time = time.time()
            logger.info(f'[生成回答] 开始处理, query: {query}, subject_filter: {subject_filter}')

            # 1. 对query进行二分类
            category = self.query_classifier.predict_category(query)
            logger.info(f'[生成回答] query分类结果: {category}')

            # 2. 对于通用问题，直接调用模型回答
            if category == '通用知识':
                logger.info(f'[生成回答] 通用知识分支，直接调用大模型回答')
                rag_prompt = self.prompts.rag_prompt()
                prompt_format = rag_prompt.format(question=query, phone=config.CUSTOMER_SERVICE_PHONE, context='', history='')
                answer = self._call_model(prompt_format)
                elapsed = time.time() - start_time
                if answer:
                    logger.info(f'[生成回答] 通用知识回答生成成功, 总耗时: {elapsed:.2f}s')
                else:
                    logger.warning(f'[生成回答] 通用知识回答生成失败, 总耗时: {elapsed:.2f}s')
                return answer

            # 3. 对于专业问题，调用策略检索
            strategy = self.strategy_selector.select_strategy(query)
            logger.info(f'[生成回答] 专业知识分支，使用策略: {strategy}')

            # TODO 作用：输入query返回父块
            chunks = self.retrieve_and_merge(query, subject_filter=subject_filter, strategy=strategy)
            logger.info(f'[生成回答] 检索到 {len(chunks)} 个文档块作为上下文')

            context = '\n\n'.join([chunk.page_content for chunk in chunks])

            rag_prompt = self.prompts.rag_prompt()
            prompt_format = rag_prompt.format(question=query, phone=config.CUSTOMER_SERVICE_PHONE, context=context, history='')

            answer = self._call_model(prompt_format)
            elapsed = time.time() - start_time
            if answer:
                logger.info(f'[生成回答] 专业知识回答生成成功, 总耗时: {elapsed:.2f}s')
            else:
                logger.warning(f'[生成回答] 专业知识回答生成失败, 总耗时: {elapsed:.2f}s')
            return answer
        except Exception as e:
            logger.error(f'[生成回答] 处理异常: {e}', exc_info=True)
            return None

if __name__ == '__main__':
    system = RagSystem()
    answer = system.generate_answer('python大模型和JAVA大模型的课程有什么区别')
    print(answer)
