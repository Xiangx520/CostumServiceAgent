"""Tools exposed for integrations; services are initialized only when called."""
from functools import lru_cache
from langchain_core.tools import tool
from RAG.rag_service import RagSummarizeService


@lru_cache(maxsize=1)
def get_rag_service():
    return RagSummarizeService()


@tool(description='从学业事务知识库检索资料并回答问题')
def rag_summarize(query: str) -> str:
    return get_rag_service().rag_summarize(query)


@tool(description='根据学业事务知识库生成咨询报告')
def generate_report(query: str) -> str:
    return get_rag_service().answer(query, report=True)['answer']
