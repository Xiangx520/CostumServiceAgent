"""Command line entry point: python main.py --help."""
import argparse
import json
import sys
from pathlib import Path


def nonempty(value):
    if not value.strip():
        raise argparse.ArgumentTypeError('问题不能为空。')
    return value.strip()


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('必须为正整数。')
    return number


def build_parser():
    parser = argparse.ArgumentParser(description='吉林大学本科生学业事务咨询助手（命令行）')
    commands = parser.add_subparsers(dest='command')
    ingest = commands.add_parser('ingest', help='导入或更新 TXT、Markdown、PDF 知识库')
    ingest.add_argument('--data-path', help='文档目录，相对路径以项目根目录为基准')
    commands.add_parser('status', help='显示知识库状态和密钥配置情况')
    for name, help_text in [('ask', '单次问答'), ('search', '检索原文'), ('report', '生成咨询报告')]:
        child = commands.add_parser(name, help=help_text)
        child.add_argument('query', type=nonempty, help='问题或报告主题')
        child.add_argument('--top-k', type=positive, help='检索片段数量')
        child.add_argument('--json', action='store_true', help='输出 JSON')
        if name == 'report':
            child.add_argument('--output', help='保存 Markdown 报告的路径')
    chat = commands.add_parser('chat', help='多轮交互，支持 /help、/clear、/save、/load、/exit')
    chat.add_argument('--top-k', type=positive)
    chat.add_argument('--history', help='启动时恢复指定 JSON 会话文件')
    return parser


def render(result):
    text = result['answer']
    if result['sources']:
        text += '\n\n检索来源：'
        for source in result['sources']:
            page = source.get('page')
            location = f'，第 {page + 1} 页' if isinstance(page, int) else ''
            text += f'\n【参考资料{source["number"]}】{source.get("source", "未知")}{location}'
    return text


def print_chunks(sources, stream=None):
    stream = sys.stdout if stream is None else stream
    print('\n检索到的 chunks：', file=stream)
    if not sources:
        print('（未检索到资料）', file=stream)
    for source in sources:
        page = source.get('page')
        location = f'，第 {page + 1} 页' if isinstance(page, int) else ''
        print(f'【参考资料{source["number"]}】{source.get("source", "未知")}{location}\n'
              f'{source["content"]}\n', file=stream)
    stream.flush()


def run_chat(agent):
    print('学业咨询助手。输入 /help 查看命令，/exit 退出。')
    while True:
        try:
            query = input('\n你：').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n会话结束。')
            return
        if not query:
            continue
        command, _, argument = query.partition(' ')
        if command in ('/exit', '/quit'):
            return
        try:
            if command == '/help':
                print('/clear 清空上下文；/save 路径 保存会话；/load 路径 恢复会话；/exit 退出')
            elif command == '/clear':
                agent.clear()
                print('上下文已清空。')
            elif command in ('/save', '/load'):
                if not argument.strip():
                    print('请在命令后指定文件路径。')
                    continue
                getattr(agent, command[1:])(argument.strip().strip('"'))
                print('会话已保存。' if command == '/save' else '会话已恢复。')
            elif command.startswith('/'):
                print('未知命令，输入 /help 查看帮助。')
            else:
                print('\n助手：' + render(agent.ask(query, on_retrieved=print_chunks)))
        except KeyboardInterrupt:
            print('\n本次请求已中断。')
        except Exception as exc:
            from utils.logger_handler import logger
            logger.debug('交互请求失败', exc_info=True)
            print(f'操作失败：{exc}', file=sys.stderr)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(['chat'])
    try:
        from RAG.vector_store import VectorStoreService
        from utils.config_handler import chroma_conf
        if args.command == 'status':
            import os
            from utils.path_tool import get_abs_path
            # No model or API key is needed to inspect the local database.
            from langchain_chroma import Chroma
            db = Chroma(collection_name=chroma_conf['collection_name'],
                        persist_directory=get_abs_path(chroma_conf['persist_directory']))
            print(f'知识库目录：{get_abs_path(chroma_conf["persist_directory"])}')
            print(f'文档片段：{len(db.get(include=[])["ids"])}')
            print('DASHSCOPE_API_KEY：' + ('已配置' if os.getenv('DASHSCOPE_API_KEY') else '未配置'))
        elif args.command == 'ingest':
            stats = VectorStoreService().load_document(args.data_path)
            print(f'导入 {stats["loaded"]} 个文件，跳过 {stats["skipped"]} 个未变化文件，'
                  f'空文档 {stats["empty"]} 个，新增片段 {stats["chunks"]} 个。')
        elif args.command == 'search':
            docs = VectorStoreService().get_retriever(args.top_k).invoke(args.query)
            result = {'answer': '检索结果' if docs else '未找到资料，请先运行 python main.py ingest。',
                      'sources': [dict(number=i, content=d.page_content, **d.metadata)
                                  for i, d in enumerate(docs, 1)]}
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                for source in result['sources']:
                    print(f'【参考资料{source["number"]}】{source.get("source", "未知")}\n{source["content"]}\n')
                if not docs:
                    print(result['answer'])
        else:
            from RAG.rag_service import RagSummarizeService
            from agent.service import CustomerServiceAgent
            agent = CustomerServiceAgent(rag=RagSummarizeService(top_k=args.top_k))
            if args.command == 'chat':
                if args.history:
                    agent.load(args.history)
                run_chat(agent)
            else:
                result = agent.ask(
                    args.query, report=args.command == 'report',
                    on_retrieved=lambda sources: print_chunks(
                        sources, stream=sys.stderr if args.json else sys.stdout))
                text = render(result)
                if args.command == 'report' and args.output:
                    target = Path(args.output).expanduser()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text + '\n', encoding='utf-8')
                print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else text)
        return 0
    except KeyboardInterrupt:
        print('\n操作已中断。', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'操作失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
