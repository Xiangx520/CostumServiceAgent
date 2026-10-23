"""Retrieval grounded answers and reports."""
import sys
from pathlib import Path

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from RAG.vector_store import VectorStoreService
from model.factory import get_chat_model
from utils.prompts_loader import load_rag_prompts, load_report_prompts, load_system_prompts


class RagSummarizeService:
    def __init__(self, vector_store=None, model=None, top_k=None):
        self.vector_store = vector_store if vector_store is not None else VectorStoreService()
        self.retriever = self.vector_store.get_retriever(top_k)
        self.retriver = self.retriever
        self._model = model

    def retrieve_docs(self, query):
        if not query.strip():
            raise ValueError('问题不能为空。')
        return self.retriever.invoke(query)

    retriver_docs = retrieve_docs

    def answer(self, query, history=None, report=False, on_retrieved=None):
        query = query.strip()
        if not query:
            raise ValueError('问题不能为空。')
        history = history or []
        recent = [m['content'] for m in history if m['role'] == 'user'][-2:]
        docs = self.retrieve_docs('\n'.join(recent + [query]))
        sources = [dict(number=i, content=d.page_content, **d.metadata)
                   for i, d in enumerate(docs, 1)]
        if on_retrieved is not None:
            on_retrieved(sources)
        if not docs:
            return {'answer': '当前知识库没有可用参考资料，请先运行 python main.py ingest 导入文档。', 'sources': []}
        context = '\n\n'.join(
            f'【参考资料{i}】 来源：{d.metadata.get("source", "未知")} '
            f'页码（从 0 开始）：{d.metadata.get("page", "未提供")}\n{d.page_content}'
            for i, d in enumerate(docs, 1))
        template = load_report_prompts() if report else load_rag_prompts()
        prompt = ChatPromptTemplate.from_messages([
            ('system', load_system_prompts()), ('placeholder', '{history}'), ('human', template)])
        model = self._model if self._model is not None else get_chat_model()
        answer = (prompt | model | StrOutputParser()).invoke({
            'input': query, 'context': context,
            'history': [(m['role'], m['content']) for m in history]})
        return {'answer': answer, 'sources': sources}

    def rag_summarize(self, query):
        return self.answer(query)['answer']


RagSummrizeService = RagSummarizeService

if __name__ == '__main__':
    print(RagSummarizeService().rag_summarize('辅修学科需要满足什么？'))
