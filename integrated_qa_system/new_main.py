# new_main.py

# 导入时间库，用于记录处理时间
import time
import uuid

# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI

# 导入 MySQL 系统组件，用于数据库操作和搜索
from mysql_qa.retreival.bm25_search import Bm25Search
from mysql_qa.db.mysql_client import MysqlClient
from mysql_qa.cache.redis_client import RedisClient

# 导入 RAG 系统组件，用于知识库检索和答案生成
# TODO 改造rag_system
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.vector_store import VectorStore

# 导入配置和日志工具，用于系统配置和日志记录
from base.config import single_config as config
from base.logger import single_logger as logger

import pymysql

"""
需求：实现FAQ和RAG模块的结合，用户输入一个query，返回最终的答案
思路步骤：
1. 接收用户的query，记录开始时间
2. 调用BM25Search.query，得到答案和是否需要继续查询RAG系统
3. 如果得到答案，直接返回
4. 如果没有答案，并且需要继续查询RAG系统，调用RAGSystem.generate_answer得到结果
5. 统计时长
"""


class IntegratedQASystem():
    def __init__(self):
        # 初始化日志工具，用于记录系统运行信息
        self.logger = logger
        self.config = config

        # 初始化 MySQL 客户端，用于数据库操作
        self.mysql_client = MysqlClient()
        # 初始化 Redis 客户端，用于缓存管理
        self.redis_client = RedisClient()

        # 初始化 BM25 搜索模块，结合 MySQL 和 Redis
        self.faq = Bm25Search(
            mysql_client=self.mysql_client
            , redis_client=self.redis_client
        )

        try:
            # 初始化 OpenAI 客户端，连接 DashScope API
            self.client = OpenAI(api_key=config.DASHSCOPE_API_KEY,
                                 base_url=config.DASHSCOPE_BASE_URL)
        except Exception as e:
            # 记录 OpenAI 初始化失败的错误日志
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            # 抛出异常，终止初始化
            raise

        # 初始化向量存储，用于 RAG 系统的知识库管理
        self.vector_store = VectorStore()
        # 初始化 RAG 系统，传入向量存储和 DashScope API 调用函数
        self.rag = RAGSystem(
            vector_store=self.vector_store
            , llm=self.call_dashscope
        )

        # 初始化对话历史表，用于存储会话记录
        # TODO 对于有session的版本，在系统初始化的时候，进行建表操作
        self.init_conversation_table()

    def call_dashscope(self, prompt):
        """调用DashScope API生成答案（流式输出）"""
        try:
            # 创建聊天完成请求，启用流式输出
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,  # 使用配置中的语言模型
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},  # 系统提示
                    {"role": "user", "content": prompt},  # 用户输入的提示
                ],
                timeout=30,  # 设置 30 秒超时
                stream=True  # 启用流式输出
            )
            # 初始化收集流式输出的字符串
            collected_content = ""
            # 遍历流式输出的每个 chunk
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    # 获取当前 chunk 的内容
                    content = chunk.choices[0].delta.content
                    # 累积内容
                    collected_content += content
                    # 逐 token 返回，供前端实时显示(生成器返回)
                    yield content
            # 返回完整答案
            return collected_content
        except Exception as e:
            # 记录 API 调用失败的错误日志
            self.logger.error(f"LLM调用失败: {e}")
            # 返回错误信息
            return f"错误：LLM调用失败 - {e}"

    def init_conversation_table(self):
        """初始化MySQL中的conversations表，用于存储对话历史"""
        try:
            # 创建 conversations 表，包含会话 ID、问题、答案和时间戳
            self.mysql_client.cursor.execute("""
                                             CREATE TABLE IF NOT EXISTS conversations
                                             (
                                                 id          BIGINT AUTO_INCREMENT PRIMARY KEY,
                                                 session_id  VARCHAR(36)           NOT NULL,
                                                 question    TEXT                  NOT NULL,
                                                 answer      TEXT                  NOT NULL,
                                                 timestamp   DATETIME              NOT NULL,
                                                 _is_deleted BOOLEAN DEFAULT FALSE NOT NULL,
                                                 INDEX idx_session_id (session_id)
                                             )
                                             """)
            # 提交数据库事务
            self.mysql_client.connect.commit()
            # 记录表初始化成功的日志
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            # 记录表初始化失败的错误日志
            self.logger.error(f"初始化对话历史表失败: {e}")
            # 抛出异常，终止初始化
            raise

    def _fetch_recent_history(self, session_id: str) -> list:
        """获取最近5轮对话历史"""
        try:
            # 执行 SQL 查询，获取最近 5 轮对话
            self.mysql_client.cursor.execute("""
                                             SELECT question, answer
                                             FROM conversations
                                             WHERE session_id = %s
                                               and _is_deleted = FALSE
                                             ORDER BY timestamp DESC
                                             LIMIT %s
                                             """, (session_id, 5))
            # 将查询结果转换为字典列表
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            # 反转结果，按时间正序返回
            # TODO 因为ORDER BY timestamp DESC， 根据时间进行了倒排。但是，我们阅读上下文的时候，需要根据时间顺序正排，所以这里需要反转
            return history[::-1]
        except pymysql.MySQLError as e:
            # 记录查询失败的错误日志
            self.logger.error(f"获取对话历史失败: {e}")
            # 返回空列表
            return []

    def get_session_history(self, session_id: str) -> list:
        """从MySQL获取会话历史"""
        # 调用 _fetch_recent_history 获取对话历史
        return self._fetch_recent_history(session_id)

    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """更新会话历史到MySQL，保留最近5轮对话"""
        try:
            # 插入新的对话记录
            self.mysql_client.cursor.execute("""
                                             INSERT INTO conversations (session_id, question, answer, timestamp)
                                             VALUES (%s, %s, %s, NOW())
                                             """, (session_id, question, answer))
            # 获取更新后的对话历史
            history = self._fetch_recent_history(session_id)

            # TODO 对于实际生产场景，用户的行为数据就是数据资产，一般来讲我们不会进行物理删除。

            # 删除超出 5 轮的旧记录
            # TODO：复杂嵌套SQL，从最内层的括号开始看，然后逐渐向外
            # 1. 【取最近5条记录】获取最近5轮对话的ID
            # 2. 【取全集和最近5条记录差集】查询当前会话(session_id)下，id不在 获取最近5轮对话的ID以内的
            # 3. 【删除差集】删除第二步的结果
            # self.mysql_client.cursor.execute("""
            #                                  DELETE
            #                                  FROM conversations
            #                                  WHERE session_id = %s
            #                                    AND id NOT IN (SELECT id
            #                                                   FROM (SELECT id
            #                                                         FROM conversations
            #                                                         WHERE session_id = %s
            #                                                         ORDER BY timestamp DESC
            #                                                         LIMIT %s) AS sub)
            #                                  """, (session_id, session_id, 5))
            # 提交事务
            self.mysql_client.connect.commit()
            # 记录更新成功的日志
            self.logger.info(f"会话 {session_id} 历史更新成功")
            # 返回更新后的历史
            return history
        except pymysql.MySQLError as e:
            # 记录数据库操作失败的错误日志
            self.logger.error(f"更新会话历史失败: {e}")
            # 回滚事务
            self.mysql_client.connect.rollback()
            # 抛出异常
            raise
        except Exception as e:
            # 记录意外错误的日志
            self.logger.error(f"更新会话历史意外错误: {e}")
            # 回滚事务
            self.mysql_client.connect.rollback()
            # 抛出异常
            raise

    def clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        try:
            # 删除指定 session_id 的所有对话记录
            # old_sql = """
            #      DELETE
            #      FROM conversations
            #      WHERE session_id = %s \
            #      """

            new_sql = """
                      update conversations
                      set _is_deleted = True
                      WHERE session_id = %s \
                      """
            self.mysql_client.cursor.execute(new_sql, (session_id,))
            # 提交事务
            self.mysql_client.connect.commit()
            # 记录清除成功的日志
            self.logger.info(f"会话 {session_id} 历史已清除")
            # 返回 True 表示成功
            return True
        except pymysql.MySQLError as e:
            # 记录清除失败的错误日志
            self.logger.error(f"清除会话历史失败: {e}")
            # 回滚事务
            self.mysql_client.connect.rollback()
            # 返回 False 表示失败
            return False

    """
    需求：实现FAQ和RAG模块的结合，支持多轮对话和流式输出，用户输入一个query，返回最终的答案
    思路步骤：
    1. 接收用户的query，记录开始时间
    2. 通过session_id获取最近的对话记录 , 如果没有session_id
    3. 调用BM25Search.query，得到答案和是否需要继续查询RAG系统
    4. 如果得到答案，直接返回。 记录对话记录到session_id中
    5. 如果没有答案，并且需要继续查询RAG系统，调用RAGSystem.generate_answer流式得到结果，并记录到session_id中
    6. 统计时长
    """

    def query(self, query, session_id=None, subject_filter=None):
        # 1. 接收用户的query，记录开始时间
        start_time = time.time()

        history = self._fetch_recent_history(session_id=session_id) if session_id else []

        # 2. 调用BM25Search.query，得到答案和是否需要继续查询RAG系统
        answer, need_rag = self.faq.search(query, threshold=0.85)
        # 3. 如果得到答案，直接返回
        if answer:
            end_time = time.time()
            duration = end_time - start_time
            logger.info("在FAQ模块获取到了答案， 执行时间: {}".format(duration))
            # 如果在FAQ模块中得到了答案，记录到对话历史
            if session_id:
                self.update_session_history(session_id=session_id, question=query, answer=answer)
            yield answer, True

        logger.info(f"在FAQ模块中未能找到可靠的答案，问题：{query}")

        # 4. 如果没有答案，并且需要继续查询RAG系统，调用RAGSystem.generate_answer得到结果
        if need_rag:
            logger.info(f"尝试查询RAG模块，问题：{query}")
            collected_answer = ''
            # generate_answer返回的内容有两种格式：生成器、字符串
            for token in self.rag.generate_answer(query, subject_filter=subject_filter, history=history):
                # 累积答案
                collected_answer += token
                # 逐 token 返回，标记为部分答案
                yield token, False
            # 如果在RAG系统重获取到了答案，记录到对话历史
            if session_id:
                self.update_session_history(session_id=session_id, question=query, answer=collected_answer)
            # 5. 统计时长
            end_time = time.time()
            duration = end_time - start_time
            logger.info("在RAG系统中获取到了答案， 执行时间: {}".format(duration))
            yield '', True
        else:
            end_time = time.time()
            duration = end_time - start_time
            logger.info("未能查询到对应的答案: {}".format(duration))
            yield f"用户输入的查询非法，未能查询到对应的答案。请联系客服：{config.CUSTOMER_SERVICE_PHONE}", True

def main():
    # 定义主函数，提供命令行交互界面
    qa_system = IntegratedQASystem()  # 初始化问答系统
    # 生成唯一会话 ID
    session_id = str(uuid.uuid4())
    # 打印欢迎信息
    print("\n欢迎使用集成问答系统！")
    # 打印会话 ID
    print(f"会话ID: {session_id}")
    # 打印支持的资源类别
    print(f"支持的资源类别：{qa_system.config.VALID_SOURCES}")
    # 提示用户输入查询或退出
    print("输入查询进行问答，输入 'exit' 退出。")
    try:
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
            source_filter = input(f"请输入资源类别 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                # 如果资源过滤无效，记录警告日志
                logger.warning(f"无效的资源类别 '{source_filter}'，将不过滤")
                # 设置为空，忽略过滤
                source_filter = None
            # 打印答案提示
            print("\n答案: ", end="", flush=True)
            # 初始化累积答案的字符串
            answer = ""
            # 迭代 query 方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    # 仅当 token 非空时打印
                    print(token, end="", flush=True)
                    # 累积答案
                    answer += token
                if is_complete:
                    # 如果是完整答案或流结束，换行并退出循环
                    print()
                    break
            # 打印对话历史
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                # 按顺序打印历史记录
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        # 记录系统错误日志
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭 MySQL 连接
        qa_system.mysql_client.close()

if __name__ == "__main__":
    # 如果脚本作为主程序运行，调用 main 函数
    main()