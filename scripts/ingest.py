"""scripts/ingest.py —— 知识库建库 + 检索一体化脚本。

用法（在项目根目录执行）：
    # 1) 建库：加载目录 -> 父子块切分 -> 写入 Milvus
    python scripts/ingest.py --dir rag_qa/data/ai_data
    # 2) 建库后立刻检索
    python scripts/ingest.py --dir rag_qa/data/ai_data --query "大模型应用开发需要学习什么"

说明：
- 目录名需形如 `{资源名}_data`（如 ai_data -> subject=ai），与
  `document_processor.load_documents_from_directory` 的约定保持一致。
- 逻辑复用 `rag_qa` 下的既有函数，不额外引入业务规则。
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def _project_root() -> pathlib.Path:
    """同时兼容项目根（base/ + rag_qa/）和 integrated_qa_system/ 两种布局。"""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "integrated_qa_system" / "base").is_dir():
            return parent / "integrated_qa_system"
        if (parent / "base").is_dir() and (parent / "rag_qa").is_dir():
            return parent
    raise RuntimeError("找不到项目根目录（需要 base/ 与 rag_qa/）")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import single_config as config  # noqa: E402
from base.logger import single_logger as logger  # noqa: E402
from rag_qa.core.document_processor import process_documents  # noqa: E402
from rag_qa.core.vector_store import VectorStore  # noqa: E402


def ingest_directory(
    directory: str,
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
    overlap: int | None = None,
    batch_size: int = 1000,
    store: VectorStore | None = None,
):
    """加载目录 -> 切父子块 -> 写向量库，返回 (store, child_chunks, parent_ids)。"""
    directory = str(directory)
    if not pathlib.Path(directory).is_dir():
        raise FileNotFoundError(f"目录不存在: {directory}")

    # 1. 加载 + 切分（父块 1200 / 子块 300 来自 config.ini，可被参数覆盖）
    child_chunks = process_documents(
        directory,
        parent_chunk_size=parent_chunk_size or config.PARENT_CHUNK_SIZE,
        child_chunk_size=child_chunk_size or config.CHILD_CHUNK_SIZE,
        overlap_size=overlap if overlap is not None else config.CHUNK_OVERLAP,
    )
    parent_ids = {d.metadata["parent_id"] for d in child_chunks}
    logger.info(f"切分完成：子块 {len(child_chunks)} 个，父块 {len(parent_ids)} 个")

    # 2. 写库（add_documents 内部按子块内容 md5 生成主键，重复导入不会翻倍）
    store = store or VectorStore()
    store.add_documents(child_chunks, batch_size=batch_size)

    # 3. 用 Milvus 里的真实条数做校验
    stored = store.milvus_client.query(store.milvus_collection_name, limit=100000)
    logger.info(f"入库完成：Milvus 当前 {len(stored)} 条子块")
    return store, child_chunks, parent_ids


def search(store: VectorStore, query: str, k: int | None = None, m: int | None = None):
    """检索并返回父块列表（复用 hybrid_search_with_rerank）。"""
    return store.hybrid_search_with_rerank(
        query,
        k=k or config.RETRIEVAL_K,
        m=m or config.CANDIDATE_M,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="建库 + 检索")
    parser.add_argument("--dir", required=True, help="文档目录，目录名需形如 ai_data")
    parser.add_argument("--query", default="", help="建库后要检索的问题；为空则只建库")
    parser.add_argument("--parent-chunk", type=int, default=None)
    parser.add_argument("--child-chunk", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--k", type=int, default=None, help="粗排返回数量")
    parser.add_argument("--m", type=int, default=None, help="精排后保留的父块数")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store, child_chunks, parent_ids = ingest_directory(
        args.dir,
        parent_chunk_size=args.parent_chunk,
        child_chunk_size=args.child_chunk,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    print(f"[ingest] 目录={args.dir}")
    print(f"[ingest] 子块数={len(child_chunks)} 父块数={len(parent_ids)}")
    for pid in sorted(parent_ids):
        pieces = sum(1 for d in child_chunks if d.metadata["parent_id"] == pid)
        print(f"[ingest]   {pid}: {pieces} 个子块")

    if args.query:
        docs = search(store, args.query, k=args.k, m=args.m)
        print(f"[query] {args.query} -> {len(docs)} 个父块")
        for i, doc in enumerate(docs, 1):
            print(f"[query]   [{i}] {doc.metadata['parent_id']} {doc.page_content[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
