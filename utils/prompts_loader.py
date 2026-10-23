from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from pathlib import Path




def load_system_prompts():
    try:
        system_prompts_path = get_abs_path(prompts_conf['system_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有system_prompts_path配置项")
        raise e

    try:
        return Path(system_prompts_path).read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错, {str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompts_path = get_abs_path(prompts_conf['rag_summrize_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summrize_prompt_path配置项")
        raise e

    try:
        return Path(rag_prompts_path).read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错, {str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompts_path = get_abs_path(prompts_conf['report_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return Path(report_prompts_path).read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错, {str(e)}")
        raise e


if __name__ == "__main__":
    print(load_report_prompts())
