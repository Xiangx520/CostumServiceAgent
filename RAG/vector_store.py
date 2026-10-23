"""Persistent, incremental knowledge base."""
import sys
import hashlib
from pathlib import Path

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from model.factory import get_embedding_model
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.file_handler import text_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self, embedding_model: Embeddings | None = None):
        self.vector_store = Chroma(
            collection_name=chroma_conf['collection_name'],
            embedding_function=embedding_model if embedding_model is not None else get_embedding_model(),
            persist_directory=get_abs_path(chroma_conf['persist_directory']),
        )
        self.vectors_stroe = self.vector_store  # Compatibility with existing callers.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf['chunk_size'], chunk_overlap=chroma_conf['chunk_overlap'],
            separators=chroma_conf['separators'], length_function=len,
        )
        self.spliter = self.splitter

    def get_retriever(self, top_k=None):
        k = chroma_conf['topK'] if top_k is None else top_k
        if k < 1:
            raise ValueError('检索条数必须大于 0。')
        return self.vector_store.as_retriever(search_kwargs={'k': k})

    get_retriver = get_retriever

    def count(self):
        return len(self.vector_store.get(include=[])['ids'])

    def load_document(self, data_path=None):
        directory = get_abs_path(data_path or chroma_conf['data_path'])
        paths = listdir_with_allowed_type(directory, tuple(
            '.' + s.lower().lstrip('.') for s in chroma_conf['allow_knowledge_file_type']))
        if paths is None:
            raise FileNotFoundError(f'知识库数据目录不存在：{directory}')
        stats = dict(loaded=0, skipped=0, empty=0, chunks=0)
        failed = []
        for path in paths:
            new_ids = []
            committed = False
            try:
                digest = get_file_md5_hex(path)
                if digest is None:
                    raise OSError(f'无法读取文件：{path}')
                source = str(Path(path).resolve())
                existing = self.vector_store.get(where={'source': source}, include=['metadatas'])
                if existing['ids'] and all(m.get('content_hash') == digest for m in existing['metadatas']):
                    stats['skipped'] += 1
                    continue
                docs = pdf_loader(path) if path.lower().endswith('.pdf') else text_loader(path)
                chunks = self.splitter.split_documents([d for d in docs if d.page_content.strip()])
                if not chunks:
                    if existing['ids']:
                        self.vector_store.delete(ids=existing['ids'])
                    stats['empty'] += 1
                    continue
                source_key = hashlib.sha256(source.encode('utf-8')).hexdigest()[:24]
                ids = [f'{source_key}:{digest}:{i}' for i in range(len(chunks))]
                for i, doc in enumerate(chunks):
                    doc.metadata.update(source=source, content_hash=digest, chunk_index=i)
                for start in range(0, len(chunks), 10):
                    batch_ids = ids[start:start + 10]
                    new_ids.extend(batch_ids)
                    self.vector_store.add_documents(chunks[start:start + 10], ids=batch_ids)
                obsolete = [i for i in existing['ids'] if i not in ids]
                if obsolete:
                    self.vector_store.delete(ids=obsolete)
                committed = True
                stats['loaded'] += 1
                stats['chunks'] += len(chunks)
                # Legacy audit file; import state is determined from the database.
                md5_path = Path(get_abs_path(chroma_conf['md5_hex_store']))
                md5_path.parent.mkdir(parents=True, exist_ok=True)
                known = set(md5_path.read_text(encoding='utf-8').splitlines()) if md5_path.exists() else set()
                if digest not in known:
                    with md5_path.open('a', encoding='utf-8') as stream:
                        stream.write(digest + '\n')
                logger.info('已导入 %s（%s 个片段）', path, len(chunks))
            except Exception:
                if new_ids and not committed:
                    self.vector_store.delete(ids=new_ids)
                failed.append(path)
                logger.exception('文档导入失败：%s', path)
        if failed:
            raise RuntimeError(f'{len(failed)} 个文件加载失败，请查看日志：' + '、'.join(failed))
        return stats


VectorStore = VectorStoreService

if __name__ == '__main__':
    print(VectorStoreService().load_document())
