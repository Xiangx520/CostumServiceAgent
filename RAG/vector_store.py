import os
import sys

# 同时支持直接运行本文件和 python -m RAG.vector_store。
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


from model.factory import get_embedding_model
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.file_handler import text_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self, embedding_model: Embeddings | None = None):
        if embedding_model is None:
            embedding_model = get_embedding_model()
        self.vectors_stroe = Chroma(
            collection_name=chroma_conf['collection_name'],
            embedding_function=embedding_model,
            persist_directory=get_abs_path(chroma_conf['persist_directory']),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf['chunk_size'],
            chunk_overlap=chroma_conf['chunk_overlap'],
            separators=chroma_conf['separators'],
            length_function=len
        )


    def get_retriver(self):
        return self.vectors_stroe.as_retriever(search_kwargs={"k": chroma_conf['topK']})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        :return: None
        """

        def check_md5_hex(md5_for_check: str) -> bool:
            md5_hex_store = get_abs_path(chroma_conf['md5_hex_store'])
            if not os.path.exists(md5_hex_store):
                return False

            with open(md5_hex_store, 'r', encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

            return False

        def save_md5_hex(md5_for_check: str):
            md5_hex_store = get_abs_path(chroma_conf['md5_hex_store'])
            os.makedirs(os.path.dirname(md5_hex_store), exist_ok=True)
            with open(md5_hex_store, 'a', encoding="utf-8") as f:
                f.write(md5_for_check + '\n')

        def get_file_documents(read_path: str):
            if read_path.lower().endswith(('.txt', '.md')):
                return text_loader(read_path)
            elif read_path.lower().endswith('.pdf'):
                return pdf_loader(read_path)
            return []


        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf['data_path']),
            tuple('.' + suffix.lower().lstrip('.') for suffix in chroma_conf['allow_knowledge_file_type']),
        )

        if allowed_files_path is None:
            raise FileNotFoundError(f"知识库数据目录不存在：{get_abs_path(chroma_conf['data_path'])}")

        failed_files = []
        for path in allowed_files_path:
            try:
                md5_hex = get_file_md5_hex(path)
                if md5_hex is None:
                    raise OSError(f"无法计算文件 MD5：{path}")
                if check_md5_hex(md5_hex):
                    logger.info(f"[load_document]{path}内容已经存在于知识库，pass")
                    continue
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[load_document]{path}中没有有效文本，pass")
                    continue
                chunks: list[Document] = self.spliter.split_documents(documents)
                if not chunks:
                    logger.warning(f"[load_document]{path}分片后没有有效文本内容，pass")
                    continue
                # 固定 ID，让入库成功但 MD5 记录失败后的重试也不会重复写入。
                # 每批最多 10 条，兼容当前 DashScope 文本向量模型的批量限制。
                batch_size = 10
                for start in range(0, len(chunks), batch_size):
                    batch = chunks[start:start + batch_size]
                    self.vectors_stroe.add_documents(
                        batch,
                        ids=[f"{md5_hex}:{index}" for index in range(start, start + len(batch))],
                    )
                save_md5_hex(md5_hex)
                logger.info(f"[load_document]{path}内容加载成功")
            except Exception as e:
                failed_files.append(path)
                logger.error(f"[load_document]{path}加载失败, {str(e)}", exc_info=True)

        if failed_files:
            raise RuntimeError(f"{len(failed_files)} 个文件加载失败，请查看日志。")



if __name__ == '__main__':
    vector_store = VectorStoreService()
    vector_store.load_document()
    retriver = vector_store.get_retriver()
    res = retriver.invoke("学业")
    if not res:
        raise RuntimeError("检索结果为空，请检查知识库数据和配置。")
    for chunk in res:
        print(chunk.page_content)
        print('-' * 20)
