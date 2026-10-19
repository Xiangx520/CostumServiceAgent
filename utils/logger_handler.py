from datetime import datetime
import logging
import os


from utils.path_tool import get_abs_path



# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")
# 确保日志目录存在
os.makedirs(LOG_ROOT, exist_ok=True)



# 日志格式配置:
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s | "                    # 输出时间
    "%(levelname)-8s | "                # 日志名
    "%(name)-24s | "                    # 级别: error info debug
    "%(filename)-20s:%(lineno)4d | "    # 文件名及行号
    "%(message)s"                       # 正文
)


def get_logger(
        name: str = "agent_logger",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # file handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


# 快捷获取日志管理器
logger = get_logger()



if __name__ == "__main__":
    logger.info("info logger")
    logger.warning("warning logger")
    logger.error("error logger")
    logger.debug("debug logger")