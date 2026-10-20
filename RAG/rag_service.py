import os
import sys

# 直接运行本文件时，将项目根目录加入模块搜索路径。
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from RAG.vector_store import VectorStoreService
from utils.prompts_loader import load_rag_prompts
from model.factory import get_chat_model


class RagSummrizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriver = self.vector_store.get_retriver()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_chat_model()
        self.chain = self._init_chain()


    def _init_chain(self):
        return self.prompt_template | self.model | StrOutputParser()


    def retriver_docs(self, query: str) -> list[Document]:
        return self.retriver.invoke(query)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriver_docs(query)

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            context += f"【参考资料{counter}】: <正文>: {doc.page_content} | <元数据>: {doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )



if __name__ == "__main__":
    service = RagSummrizeService()
    print(service.rag_summarize("辅修学科需要满足什么？"))




