# core/strategy_selector.py 源码
# 导入时间模块
import time
# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate
# 导入 LangChain 消息类型
from langchain_core.messages import SystemMessage, HumanMessage
# 导入 LangChain ChatOpenAI（兼容 OpenAI 接口的模型）
from langchain_openai import ChatOpenAI
# 导入日志和配置
from base.config import single_config as config
from base.logger import single_logger as logger

class StrategySelector:
    def __init__(self):
        """
        需求：初始化策略选择器实例
        思路步骤：
        1. 创建OpenAI客户端，配置API密钥和基础URL[6](@ref)
        2. 获取策略选择提示模板（包含四种策略的详细说明）
        3. 完成客户端和模板的初始化
        """
        # 初始化 LangChain ChatOpenAI 客户端（兼容 DashScope 接口）
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            temperature=0.1, # 设置低温度值保证输出确定性
        )
        # 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()

    def _get_strategy_prompt(self):
        """
            需求：定义策略选择提示模板
            思路步骤：
            1. 创建多行模板字符串，明确四种策略：
               - 直接检索：查询意图明确时使用
               - HyDE：处理抽象查询时使用
               - 子查询检索：复杂多实体查询时使用
               - 回溯问题检索：简化复杂查询时使用[6](@ref)
            2. 包含各策略的示例场景说明
            3. 声明query作为唯一输入变量
        """
        #   定义私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
               你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

               以下是几种检索增强策略及其适用场景：

               1.  **直接检索：**
                   * 描述：对用户查询直接进行检索，不进行任何增强处理。
                   * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                       * 示例：
                           * 查询：AI 需要学习哪些知识？
                           * 策略：直接检索
                       * 查询：JAVA 智能体是什么？
                           * 策略：直接检索
               2.  **假设问题检索：**
                   * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                   * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                       * 示例：
                           * 查询：人工智能在教育领域的应用有哪些？
                           * 策略：假设问题检索
               3.  **子查询检索：**
                   * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                   * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                       * 示例：
                           * 查询：比较 Milvus 和 Faiss 的优缺点。
                           * 策略：子查询检索
               4.  **回溯问题检索：**
                   * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                   * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                       * 示例：
                           * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                           * 策略：回溯问题检索

               根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
               """
            ,
            input_variables=["query"],
        )

    def call_dashscope(self, prompt):
        """
           需求：调用大模型API获取策略决策结果
           思路步骤：
           1. 构造包含系统角色和用户查询的请求消息
           2. 发送API请求
           3. 解析API响应，提取策略名称
           4. 异常处理：记录错误日志并返回默认策略"直接检索"
       """
        # 使用 LangChain ChatOpenAI 调用大模型
        try:
            # 构建 LangChain 消息列表
            messages = [
                SystemMessage(content="你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"),
                HumanMessage(content=prompt),
            ]
            # 记录开始时间
            start_time = time.time()
            # 调用 LLM 获取响应
            response = self.llm.invoke(messages)
            # 计算耗时
            elapsed_time = time.time() - start_time
            # 记录模型调用耗时
            logger.info(f"策略选择模型调用耗时: {elapsed_time:.2f}s")
            # 返回响应内容
            return response.content if response and response.content else "直接检索"
        except Exception as e:
            # 记录 API 调用失败
            logger.error(f"LangChain LLM 调用失败: {e}")
            # 默认返回直接检索
            return "直接检索"

    #   定义方法，选择检索策略
    def select_strategy(self, query):
        """
            需求：根据用户查询选择最佳检索策略
            思路步骤：
            1. 使用提示模板格式化用户查询
            2. 调用大模型API获取策略决策
            3. 记录策略选择日志
            4. 返回策略名称字符串
        """
        #   调用 LLM 获取检索策略
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy

if __name__ == '__main__':
    selector = StrategySelector()
    strategy = selector.select_strategy(query="比较mysql和postgresql的优缺点")
    print(strategy)