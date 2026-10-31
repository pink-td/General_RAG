# 工业通用 RAG 智能问答系统

面向工业领域的智能问答系统：采用 **FAQ（MySQL + Redis）+ RAG（Milvus）双引擎** 架构，
高频问题走 BM25 精确匹配与缓存、秒级返回；复杂问题自动路由到 RAG 链路，经
**BGE-M3 混合检索（Dense + Sparse）** 与 **BGE-Reranker** 精排后交给大模型生成，
并通过 **RAGAS** 做自动化评估，缓解专业领域大模型的「幻觉」与「知识滞后」问题。


## 功能

- **双引擎问答**：FAQ 命中走 MySQL + Redis 缓存，未命中自动降级到 RAG 检索。
- **混合检索**：BGE-M3 同时产出语义向量与稀疏向量，在 Milvus 中混合召回后统一重排。
- **父子块切分**：子块用于精细检索，命中后回溯父块作为上下文，兼顾精度与上下文完整性。
- **多策略检索**：支持直接检索、HyDE（假设答案）、子查询拆分、问题回溯等策略，按问题类型动态选择。
- **意图路由**：微调 BERT 做问题分类，区分高频/简单问题与复杂/长尾问题。
- **流式输出**：FastAPI + WebSocket 流式返回答案，支持多轮对话与会话历史。
- **评估闭环**：RAGAS 计算 Faithfulness、Answer Relevancy 等指标，支持回归对比。
- **容器化部署**：Docker Compose 编排应用与依赖，一键启动。

## 技术更迭

- **缺陷修复**：定位并修复流式回答在「答案结束即关闭」场景下丢字的问题（发送顺序 + 生成器
  生命周期两处）；修复 Redis 缓存数据类型不一致（bytes/str 混用）导致的判断失效；
  修复复杂提问策略（子查询 / HyDE）在部分路径下未生效、检索来源过滤不生效等问题，
  并补充了对应的回归用例。
- **建库脚本**：补写 `scripts/ingest.py`，把「加载目录 → 父子块切分 → 写入 Milvus → 立即检索」
  串成一条命令，便于复现与验收。
- **评估与加固**：接入 RAGAS 评估脚本并保存基线结果用于回归对比；补充单元测试、
  精简依赖、完善 Dockerfile 与 compose 的配置外置化。
- **可观测性**：统一日志级别与关键链路埋点，便于定位检索/生成阶段的耗时与失败原因。

## 架构

```
integrated_qa_system/
├── app.py                    FastAPI 入口（HTTP + WebSocket）
├── new_main.py               双引擎问答编排（FAQ -> RAG 降级）
├── old_main.py               早期版本编排（保留参考）
├── config.ini                MySQL / Redis / Milvus / 检索参数 / BERT 配置
├── base/                     配置加载与日志
├── mysql_qa/                 FAQ 引擎：BM25 检索、MySQL、Redis 缓存
├── rag_qa/
│   ├── core/                 RAG 编排、向量库、Prompt、问题分类、策略选择
│   ├── project_document_loaders/   PDF/Word/PPT/图片 OCR 加载
│   ├── project_text_spliter/       父子块切分
│   └── rag_assesment/        RAGAS 评估数据与脚本
└── static/                   前端页面
scripts/ingest.py             建库 + 检索脚本
others/                       数据与辅助脚本
```

问答流程：

```
用户提问
   │
   ▼
BERT 意图分类 ──► FAQ 引擎（MySQL BM25 + Redis 缓存）
   │                       │ 命中
   │                       ▼
   │                    直接返回（毫秒级）
   │ 未命中 / 复杂问题
   ▼
策略选择（直接 / HyDE / 子查询 / 回溯）
   │
   ▼
Milvus 混合检索（BGE-M3 Dense + Sparse）
   │
   ▼
BGE-Reranker 精排 ──► 父块上下文 ──► 大模型生成 ──► 流式返回
```

## 快速开始

### 1. 准备依赖服务

需要 MySQL（FAQ 知识库）、Redis（缓存）、Milvus（向量库），以及 DashScope API Key
（或其他 OpenAI 兼容的 LLM 服务）。可参考 `integrated_qa_system/docker-compose.yml`。

### 2. 配置

```bash
cd integrated_qa_system
cp .env.example .env          # 填入 DASHSCOPE_API_KEY 等
# 按实际环境修改 config.ini 中的 MySQL / Redis / Milvus 地址
```

配置优先级：环境变量 > `config.ini`。

### 3. 初始化 FAQ 与知识库

- FAQ：将 `mysql_qa/data/资源知识问答.csv` 导入 MySQL 的 `subjects` 库。
- 知识库：把文档放入形如 `rag_qa/data/ai_data/` 的目录（目录名决定 `subject` 来源），然后执行：

```bash
python ../scripts/ingest.py --dir rag_qa/data/ai_data
# 建库后直接检索验证
python ../scripts/ingest.py --dir rag_qa/data/ai_data --query "大模型应用开发需要学习什么"
```

> 首次运行会从 ModelScope 下载 BGE-M3、BGE-Reranker 等模型，请预留磁盘与网络。

### 4. 启动服务

```bash
python app.py                 # 默认监听 0.0.0.0:8080，可用 HOST / PORT 覆盖
```

或使用 Docker：

```bash
cd integrated_qa_system
docker compose up -d --build
```

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 前端页面 |
| POST | `/api/create_session` | 创建会话 |
| POST | `/api/query` | 同步问答 |
| WS | `/api/stream` | WebSocket 流式问答 |
| GET | `/api/history/{session_id}` | 会话历史 |
| DELETE | `/api/history/{session_id}` | 清空会话 |
| GET | `/api/sources` | 可用知识来源 |
| GET | `/health` | 健康检查 |

## 评估

```bash
cd integrated_qa_system
python rag_qa/rag_assesment/ragas_evaluate.py
```

评估数据见 `rag_qa/rag_assesment/rag_evaluate_data.json`，历史结果见
`ragas_evaluation_results.csv`，可用于对比改动前后的指标变化。

