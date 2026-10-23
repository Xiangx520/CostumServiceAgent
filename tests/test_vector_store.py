import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter

from RAG.vector_store import VectorStore, chroma_conf
from utils.file_handler import listdir_with_allowed_type, pdf_loader, text_loader


class LocalEmbeddings(Embeddings):
    """仅用于测试流程，生产环境仍使用 DashScope。"""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.count(word)) for word in ("学业", "注册", "辅修")] + [1.0]


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.md5_file = self.root / "state" / "md5.txt"
        self.settings = patch.dict(chroma_conf, {
            "collection_name": "vector_store_test",
            "persist_directory": str(self.root / "chroma"),
            "data_path": str(self.data),
            "md5_hex_store": str(self.md5_file),
            "allow_knowledge_file_type": ["txt", "pdf", "md"],
            "topK": 2,
        })
        self.settings.start()
        self.store = VectorStore(embedding_model=LocalEmbeddings())

    def tearDown(self):
        self.settings.stop()
        # 释放 Chroma 的 SQLite 文件句柄，允许 Windows 清理临时目录。
        self.store.vectors_stroe._client._system.stop()
        self.store.vectors_stroe._client.clear_system_cache()
        self.temp.cleanup()

    def test_load_retrieve_deduplicate_and_reopen(self):
        (self.data / "academic.txt").write_text("学业 学业 学业预警管理办法", encoding="utf-8")
        (self.data / "registration.md").write_text("注册 注册 注册管理办法", encoding="utf-8")
        self.store.load_document()
        count = self.store.vectors_stroe._collection.count()
        self.assertEqual(count, 2)
        self.assertEqual(len(self.md5_file.read_text(encoding="utf-8").splitlines()), 2)

        results = self.store.get_retriver().invoke("学业")
        self.assertEqual(len(results), 2)
        self.assertIn("学业", results[0].page_content)

        with patch.object(LocalEmbeddings, "embed_documents", side_effect=AssertionError("重复入库")):
            self.store.load_document()
        self.assertEqual(self.store.vectors_stroe._collection.count(), count)
        reopened = VectorStore(embedding_model=LocalEmbeddings())
        self.assertEqual(reopened.vectors_stroe._collection.count(), count)
        self.assertTrue(reopened.get_retriver().invoke("学业"))

    def test_failed_embedding_does_not_mark_file_as_loaded(self):
        (self.data / "academic.txt").write_text("学业预警", encoding="utf-8")
        with patch.object(LocalEmbeddings, "embed_documents", side_effect=RuntimeError("API failure")):
            with self.assertLogs("agent_logger", level="ERROR"):
                with self.assertRaises(RuntimeError):
                    self.store.load_document()
        self.assertFalse(self.md5_file.exists())
        self.store.load_document()
        self.assertEqual(self.store.vectors_stroe._collection.count(), 1)

    def test_retry_after_missing_md5_record_does_not_duplicate_chunks(self):
        (self.data / "academic.txt").write_text("学业预警", encoding="utf-8")
        self.store.load_document()
        self.md5_file.unlink()
        self.store.load_document()
        self.assertEqual(self.store.vectors_stroe._collection.count(), 1)

    def test_missing_data_directory_raises(self):
        with patch.dict(chroma_conf, {"data_path": str(self.root / "missing")}):
            with self.assertLogs("agent_logger", level="ERROR"):
                with self.assertRaises(FileNotFoundError):
                    self.store.load_document()

    def test_changed_document_replaces_old_chunks(self):
        path = self.data / 'policy.md'
        path.write_text('旧规定 学业预警', encoding='utf-8')
        self.store.load_document()
        path.write_text('新规定 注册管理', encoding='utf-8')
        stats = self.store.load_document()
        self.assertEqual(stats['loaded'], 1)
        self.assertEqual(self.store.count(), 1)
        self.assertIn('新规定', self.store.vector_store.get()['documents'][0])

    def test_same_content_in_different_sources_keeps_both(self):
        for name in ('first.md', 'second.md'):
            (self.data / name).write_text('学业预警', encoding='utf-8')
        self.store.load_document()
        self.assertEqual(self.store.count(), 2)

    def test_failed_update_preserves_previous_version(self):
        path = self.data / 'policy.md'
        path.write_text('旧规定 学业预警', encoding='utf-8')
        self.store.load_document()
        path.write_text('新规定 注册管理', encoding='utf-8')
        with patch.object(LocalEmbeddings, 'embed_documents', side_effect=RuntimeError('API failure')):
            with self.assertLogs('agent_logger', level='ERROR'):
                with self.assertRaises(RuntimeError):
                    self.store.load_document()
        self.assertEqual(self.store.count(), 1)
        self.assertIn('旧规定', self.store.vector_store.get()['documents'][0])

    def test_empty_replacement_removes_old_policy(self):
        path = self.data / 'policy.md'
        path.write_text('旧规定 学业预警', encoding='utf-8')
        self.store.load_document()
        path.write_text('', encoding='utf-8')
        self.assertEqual(self.store.load_document()['empty'], 1)
        self.assertEqual(self.store.count(), 0)

    def test_partial_batch_failure_rolls_back_and_retry_succeeds(self):
        path = self.data / 'policy.md'
        path.write_text('旧规定', encoding='utf-8')
        self.store.load_document()
        path.write_text('新规定 学业 注册 辅修。' * 300, encoding='utf-8')
        original = self.store.vector_store.add_documents
        calls = 0

        def fail_second_batch(documents, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('第二批 API 失败')
            return original(documents, **kwargs)

        with patch.object(self.store.vector_store, 'add_documents', side_effect=fail_second_batch):
            with self.assertLogs('agent_logger', level='ERROR'):
                with self.assertRaises(RuntimeError):
                    self.store.load_document()
        self.assertEqual(self.store.vector_store.get()['documents'], ['旧规定'])
        self.store.load_document()
        self.assertGreater(self.store.count(), 10)
        self.assertNotIn('旧规定', self.store.vector_store.get()['documents'])

    def test_stale_md5_file_does_not_prevent_rebuild(self):
        path = self.data / 'policy.md'
        path.write_text('学业预警', encoding='utf-8')
        self.store.load_document()
        self.store.vector_store.delete(ids=self.store.vector_store.get()['ids'])
        self.assertTrue(self.md5_file.exists())
        self.store.load_document()
        self.assertEqual(self.store.count(), 1)

    def test_file_loaders_and_extension_filter(self):
        text_path = self.data / "academic.TXT"
        text_path.write_text("中文学业管理", encoding="utf-8")
        self.assertEqual(text_loader(str(text_path))[0].page_content, "中文学业管理")
        (self.data / "directory.txt").mkdir()
        (self.data / "ignored.notxt").write_text("ignored", encoding="utf-8")
        self.assertEqual(listdir_with_allowed_type(str(self.data), (".txt",)), (str(text_path),))
        pdf_path = self.data / "blank.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(str(pdf_path))
        self.assertEqual(len(pdf_loader(str(pdf_path))), 1)

    def test_paths_do_not_depend_on_working_directory(self):
        (self.data / "academic.txt").write_text("学业预警", encoding="utf-8")
        original_cwd = os.getcwd()
        try:
            os.chdir(self.root)
            self.store.load_document()
            self.assertTrue(self.store.get_retriver().invoke("学业"))
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
