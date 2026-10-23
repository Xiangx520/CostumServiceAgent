"""Small conversational service shared by the terminal and Python callers."""
import json
from pathlib import Path

from RAG.rag_service import RagSummarizeService
from utils.config_handler import agent_conf


class CustomerServiceAgent:
    def __init__(self, rag=None, max_history_turns=None):
        self.rag = rag if rag is not None else RagSummarizeService()
        self.max_history_turns = (agent_conf.get('max_history_turns', 6)
                                  if max_history_turns is None else max_history_turns)
        if not isinstance(self.max_history_turns, int) or self.max_history_turns < 1:
            raise ValueError('max_history_turns 必须为正整数。')
        self.history = []

    def ask(self, query, report=False, on_retrieved=None):
        result = self.rag.answer(query, history=self.history, report=report,
                                 on_retrieved=on_retrieved)
        self.history.extend([{'role': 'user', 'content': query.strip()},
                             {'role': 'assistant', 'content': result['answer']}])
        self.history = self.history[-2 * self.max_history_turns:]
        return result

    def clear(self):
        self.history.clear()

    def save(self, path):
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding='utf-8')

    def load(self, path):
        messages = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
        if not isinstance(messages, list) or len(messages) % 2:
            raise ValueError('会话文件必须包含成对的用户和助手消息。')
        for i, message in enumerate(messages):
            if (not isinstance(message, dict) or message.get('role') != ('user' if i % 2 == 0 else 'assistant')
                    or not isinstance(message.get('content'), str)):
                raise ValueError('会话格式无效，仅支持交替排列的 user/assistant 文本消息。')
        self.history = [{'role': m['role'], 'content': m['content']}
                        for m in messages[-2 * self.max_history_turns:]]
