from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional
import os
from dotenv import load_dotenv

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatTongyi


from utils.config_handler import rag_conf
from utils.path_tool import get_abs_path


load_dotenv(get_abs_path('.env'))


def require_api_key():
    if not os.getenv('DASHSCOPE_API_KEY', '').strip():
        raise ValueError('未配置 DASHSCOPE_API_KEY，请复制 .env.example 为 .env 并填写 DashScope API Key。')


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass



class ChatterModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        require_api_key()
        model = ChatTongyi(model=rag_conf['chat_model_name'])
        # 当前 ChatTongyi 的自动识别列表尚未覆盖 qwen3.8-flash。
        # 初始化后指定 SDK 客户端，避免被其初始化校验覆盖。
        if rag_conf.get('chat_model_api') == 'multimodal':
            from dashscope import MultiModalConversation

            model.client = MultiModalConversation
        return model


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        require_api_key()
        return DashScopeEmbeddings(model=rag_conf['embedding_model_name'])





@lru_cache(maxsize=1)
def get_chat_model():
    return ChatterModelFactory().generator()


@lru_cache(maxsize=1)
def get_embedding_model():
    return EmbeddingsFactory().generator()


def __getattr__(name):
    # 保留原有导入方式，但仅在实际使用时创建对应模型。
    if name == 'chat_model':
        return get_chat_model()
    if name == 'embedding_model':
        return get_embedding_model()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
