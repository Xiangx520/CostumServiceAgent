import os, hashlib
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from utils.logger_handler import logger


# 获取文件的md5文件
def get_file_md5_hex(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist")
        return None

    if not os.path.isfile(file_path):
        logger.error(f"File {file_path} is not a file")
        return None

    md5_object = hashlib.md5()

    chunk_size = 4096

    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_object.update(chunk)
        md5_hex = md5_object.hexdigest()
    except Exception as e:
        logger.error(f"File {file_path} calculate md5 error when streaming, {str(e)}")
        return None

    return md5_hex


# 返回文件夹内指定后缀的文件
def listdir_with_allowed_type(dir_path: str, allowed_types: tuple[str]):
    files = []
    if not os.path.isdir(dir_path):
        logger.error(f"Directory {dir_path} is not a directory")
        return None

    for f in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, f)
        if os.path.isfile(file_path) and f.lower().endswith(allowed_types):
            files.append(file_path)

    return tuple(files)


# 加载PDF
def pdf_loader(file_path: str, password: str | None = None) -> list[Document]:
    return PyPDFLoader(file_path, password=password).load()

# 加载text
def text_loader(file_path: str) -> list[Document]:
    return TextLoader(file_path, encoding="utf-8").load()
