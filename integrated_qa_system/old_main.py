"""
需求：实现project-rag项目
思路步骤：
1. 初始化：bm25search、大模型、rag_qa_system
2. 将用户query发送给FAQ模块进行处理，得到答案和是否进入RAG模块的标识
3. 如果答案可靠，直接返回
4. 如果答案不可靠，且需要进入RAG模块，调用rag模块，得到答案
5. 如果答案不可靠，且不需要进入RAG模块，直接走兜底逻辑
"""
import time

from langchain_core.messages import SystemMessage, HumanMessage

# 导入 OpenAI 客户端，用于调用 DashScope API
from langchain_openai import ChatOpenAI

# 导入配置和日志工具，用于系统配置和日志记录
from base.logger import single_logger as logger
from base.config import single_config as config

# 导入 MySQL 系统组件，用于数据库操作和搜索
from mysql_qa.retreival.bm25_search import Bm25Search
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MysqlClient

# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa.core.rag_system import RagSystem

class IntegratedQASystem:
    # 1. 初始化：bm25search、大模型、rag_qa_system
    def __init__(self):
        # bm25search

        # 初始化配置对象，加载系统参数
        self.config = config
        # 初始化日志工具，用于记录系统运行信息
        self.logger = logger
        # 初始化 MySQL 客户端，用于数据库操作
        self.db = MysqlClient()
        # 初始化 Redis 客户端，用于缓存管理
        self.cache = RedisClient()
        # 初始化 BM25 搜索模块，结合 MySQL 和 Redis
        self.bm25_search = Bm25Search(
            mysql_client=self.db,
            redis_client=self.cache
        )

        # 初始化 OpenAI 客户端，连接 DashScope API
        # 大模型
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            temperature=0.1,
        )

        # # 初始化 RAG 系统: rag_qa_system
        self.rag_system = RagSystem()

    def _call_model(self, prompt, count_down=3):
        """
        定义调用 DashScope API 的方法，生成自然语言答案
        :param prompt:
        :param count_down:
        :return:
        """
        start_time = time.time()
        logger.debug(f'开始调用大模型，最大重试次数: {count_down}')
        for i in range(count_down):
            try:
                # 创建聊天完成请求
                messages = [
                    SystemMessage('你是一个有用的助手，能够准确无误的完成用户的命令'),
                    HumanMessage(prompt),
                ]
                # 调用 DashScope API
                result = self.llm.invoke(messages).content
                elapsed = time.time() - start_time
                logger.debug(f'大模型调用成功，第 {i + 1} 次尝试, 耗时: {elapsed:.2f}s')
                return result
            except Exception as e:
                logger.error(f'大模型调用失败，第 {i + 1}/{count_down} 次尝试，原因: {e}')
                if i < count_down - 1:
                    logger.debug(f'等待 {1} 秒后重试...')
                    time.sleep(1)
                    continue
        elapsed = time.time() - start_time
        logger.warning(f'大模型调用失败，已重试 {count_down} 次，放弃调用, 总耗时: {elapsed:.2f}s')
        return None

    def _fallback_answer(self):
        """兜底逻辑：当所有模块都无法返回有效答案时使用"""
        phone = self.config.CUSTOMER_SERVICE_PHONE
        return f'抱歉，系统繁忙，暂时无法为您解答。请拨打客服电话 {phone} 获取帮助。'

    def search(self, query, subject_filter=None):
        """
        处理用户查询，优先通过 BM25 搜索 MySQL，若无可靠答案则回退到 RAG 系统
        :param query:
        :param subject_filter:
        :return:
        """
        try:
            # 2. 将用户query发送给faq模块进行处理，得到答案和是否进入RAG模块的标识
            answer, need_rag = self.bm25_search.search(query)
        except Exception as e:
            logger.error(f'FAQ模块调用异常: {e}')
            return self._fallback_answer()

        # 3. 如果答案可靠，直接返回
        if answer:
            # 如果找到可靠答案，记录答案到日志
            logger.info(f"MySQL 答案: {answer}")
            # 返回 MySQL 答案
            return answer

        # 4. 如果答案不可靠，且需要进入RAG模块，调用rag模块，得到答案
        if need_rag:
            try:
                # 如果需要 RAG，记录回退信息到日志
                logger.info("无可靠 MySQL 答案，回退到 RAG")
                # 调用 RAG 系统生成答案，支持资源过滤
                answer = self.rag_system.generate_answer(query, subject_filter)
                # 记录 RAG 答案到日志
                logger.info(f"RAG 答案: {answer}")
            except Exception as e:
                logger.error(f'RAG模块调用异常: {e}')
                return self._fallback_answer()
            if answer:
                return answer

        # 5. 兜底逻辑
        # 如果无答案，记录信息到日志
        logger.info("未找到答案")
        return self._fallback_answer()

def main():
    # 定义主函数，提供命令行交互界面
    qa_system = IntegratedQASystem()  # 初始化问答系统
    try:
        # 打印欢迎信息
        print("\n欢迎使用集成问答系统！")
        # 打印支持的资源类别
        print(f"支持的来源: {qa_system.config.VALID_SOURCES}")
        # 提示用户输入查询或退出
        print("输入查询进行问答，输入 'exit' 退出。")
        while True:
            # 获取用户输入的查询
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 如果用户输入 exit，记录退出日志
                logger.info("退出系统")
                # 打印退出信息
                print("再见！")
                # 退出循环
                break
            # 获取用户输入的资源过滤
            source_filter = input(f"输入来源过滤 ({'/'.join(qa_system.config.VALID_SOURCES)}) (按 Enter 跳过): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                # 如果资源过滤无效，记录警告日志
                logger.warning(f"无效来源 '{source_filter}'，忽略过滤")
                # 打印无效信息，忽略过滤
                print(f"无效来源 '{source_filter}'，继续无过滤。")
                source_filter = None
            # 执行查询，获取答案
            answer = qa_system.search(query, source_filter)
            # 打印答案
            print(f"\n答案: {answer}")
    except Exception as e:
        # 记录系统错误日志
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 无论是否发生错误，关闭 MySQL 连接
        qa_system.db.close()

if __name__ == "__main__":
    # 如果脚本作为主程序运行，调用 main 函数
    main()

# if __name__ == '__main__':
#     qa_system = IntegratedQASystem()
#     qa_system_search = qa_system.search('如何创建线程安全的单例对象')
#     time.sleep(5)
#     print(qa_system_search)