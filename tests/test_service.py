import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from RAG.rag_service import RagSummarizeService
from agent.service import CustomerServiceAgent
from main import main, run_chat, print_chunks


class ServiceTests(unittest.TestCase):
    def make_rag(self, docs=None):
        store = Mock()
        store.get_retriever.return_value.invoke.return_value = docs or []
        return RagSummarizeService(vector_store=store, model=FakeListChatModel(responses=['回答【参考资料1】']))

    def test_empty_database_does_not_call_model(self):
        rag = self.make_rag()
        rag._model = Mock(side_effect=AssertionError('不应调用 API'))
        self.assertEqual(rag.answer('注册要求')['sources'], [])

    def test_answer_report_and_followup_retrieval(self):
        rag = self.make_rag([Document(page_content='注册规定', metadata={'source': 'policy.md', 'page': 0})])
        agent = CustomerServiceAgent(rag=rag, max_history_turns=1)
        result = agent.ask('注册要求')
        self.assertEqual(result['sources'][0]['number'], 1)
        agent.ask('如何办理？', report=True)
        rag.retriever.invoke.assert_called_with('注册要求\n如何办理？')
        self.assertEqual(len(agent.history), 2)

    def test_failed_request_does_not_change_history(self):
        rag = Mock()
        rag.answer.side_effect = RuntimeError('网络故障')
        agent = CustomerServiceAgent(rag=rag)
        with self.assertRaises(RuntimeError):
            agent.ask('注册要求')
        self.assertEqual(agent.history, [])

    def test_chunks_are_printed_before_model_is_called(self):
        rag = self.make_rag([Document(page_content='注册规定正文', metadata={'source': 'policy.md', 'page': 0})])
        output = io.StringIO()

        def generate(prompt):
            preview = output.getvalue()
            self.assertIn('【参考资料1】policy.md，第 1 页', preview)
            self.assertIn('注册规定正文', preview)
            self.assertNotIn('最终答案', preview)
            return '最终答案'

        rag._model = RunnableLambda(generate)
        agent = CustomerServiceAgent(rag=rag)
        with patch('builtins.input', side_effect=['注册要求', '/exit']):
            with contextlib.redirect_stdout(output):
                run_chat(agent)
        self.assertLess(output.getvalue().index('注册规定正文'), output.getvalue().index('最终答案'))

    def test_json_preview_goes_to_stderr(self):
        rag = self.make_rag([Document(page_content='注册规定正文', metadata={'source': 'policy.md'})])
        output, preview = io.StringIO(), io.StringIO()
        with patch('RAG.rag_service.RagSummarizeService', return_value=rag):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(preview):
                self.assertEqual(main(['ask', '注册要求', '--json']), 0)
        self.assertIn('answer', json.loads(output.getvalue()))
        self.assertIn('注册规定正文', preview.getvalue())

    def test_history_roundtrip_and_invalid_load_preserves_state(self):
        agent = CustomerServiceAgent(rag=self.make_rag())
        agent.ask('注册要求')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'session.json'
            agent.save(target)
            expected = agent.history.copy()
            agent.clear()
            agent.load(target)
            self.assertEqual(agent.history, expected)
            target.write_text(json.dumps([{'role': 'system', 'content': 'bad'}, {}]), encoding='utf-8')
            with self.assertRaises(ValueError):
                agent.load(target)
            self.assertEqual(agent.history, expected)

    def test_cli_help_and_blank_query(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_info:
                main(['--help'])
            self.assertEqual(exit_info.exception.code, 0)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_info:
                main(['ask', '   '])
            self.assertEqual(exit_info.exception.code, 2)

    def test_interactive_clear_and_exit(self):
        agent = CustomerServiceAgent(rag=self.make_rag())
        with patch('builtins.input', side_effect=['注册要求', '/clear', '/exit']):
            with contextlib.redirect_stdout(io.StringIO()):
                run_chat(agent)
        self.assertEqual(agent.history, [])

    def test_report_export_and_json_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'report.md'
            output = io.StringIO()
            with patch('RAG.rag_service.RagSummarizeService', return_value=self.make_rag()):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(['report', '注册规定', '--output', str(target), '--json']), 0)
            self.assertIn('answer', json.loads(output.getvalue()))
            self.assertTrue(target.read_text(encoding='utf-8').strip())


if __name__ == '__main__':
    unittest.main()
