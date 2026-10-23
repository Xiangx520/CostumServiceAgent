# 学业事务咨询助手

基于本地 Chroma 知识库与 DashScope 模型，提供吉林大学本科生学业事务检索、问答和咨询报告，全部在命令行运行。

## 安装与配置

需要 Python 3.13+ 和 uv。在项目根目录执行：

```powershell
uv sync --locked
Copy-Item .env.example .env
```

已有 `.env` 时直接编辑它，不要覆盖。填写 `DASHSCOPE_API_KEY`，也可以通过同名环境变量设置。模型名称与接口类型在 `config/rag.yml` 中配置，需确认账号拥有对应模型权限；不在日志和状态输出中展示密钥。

## 使用

将 UTF-8 编码的 `.txt`、`.md` 或可提取文本的 `.pdf` 放到 `data/` 目录，然后执行：

```powershell
uv run python main.py ingest
uv run python main.py status
uv run python main.py search "辅修申请条件" --top-k 5
uv run python main.py ask "申请辅修需要满足哪些条件？"
uv run python main.py chat
uv run python main.py report "整理转专业的申请条件与办理流程" --top-k 8 --output reports/转专业.md
```

直接运行 `uv run python main.py` 也会进入交互模式。已激活项目虚拟环境时可直接使用 `python main.py ...`。`--help` 查看命令；`ask`、`search`、`report` 支持 `--json` 输出结构化结果。报告包含检索来源，模型回答中的资料编号对应本次检索片段，不代表文件的官方编号。

问答、聊天和报告生成都会在调用模型前，先打印本次检索的 chunk 编号、来源、页码（如有）和完整正文。使用 `--json` 时，chunk 预览输出到 stderr，stdout 保持为有效 JSON。

交互命令：

- `/help`：显示帮助。
- `/clear`：清空当前上下文。
- `/save sessions/demo.json`：将当前上下文保存到 JSON。
- `/load sessions/demo.json`：恢复上下文。
- `/exit` 或 `/quit`：退出；Ctrl+C、EOF 也可以结束等待输入。

`chat --history sessions/demo.json` 可在启动时恢复会话。上下文默认保留最近 6 轮，可在 `config/agent.yml` 调整；保存的是保留的上下文，退出时不自动保存。会话和报告可能含有个人咨询内容，请自行妥善保管。

## 知识库维护

`config/chroma.yml` 控制数据目录、数据库位置、分片和默认检索数量。`ingest --data-path 路径` 可导入其他目录；相对数据路径以项目根目录为基准，报告和会话保存路径以当前工作目录为基准。

导入扫描目录第一层，支持大小写后缀。未变化文件跳过；修改后的文件会替换该来源的旧片段；失败批次清理新片段并保留旧版本。不同来源的同内容文件分别保留，避免丢失来源。数据库是导入状态的依据，旧 `md5.txt` 不会阻止空数据库重新导入。空文本替换会移除该来源的旧内容；删除或移动磁盘文件不会自动删除原来源的数据库记录。更换向量模型或分片配置时，应配置新的数据库目录并重新入库。

`status` 无需 API 密钥即可查看本地片段数量。导入和检索需要 DashScope 向量服务，问答和报告还需要聊天模型；离线测试使用本地测试向量和模拟聊天模型，不调用付费接口。扫描版 PDF 需先自行 OCR。

## 测试与排错

```powershell
uv run python -m unittest discover -s tests -v
```

缺少密钥时按提示配置 `.env`；认证、模型权限或网络失败时检查 DashScope 配置。空知识库先执行 `ingest`。导入详情记录在 `logs/`。程序错误返回非零退出码，交互模式的单次请求失败后仍可继续输入。

检索仅取有限片段，可能不覆盖完整条款；资料的现行效力仍需业务部门核实。程序用于解释和整理资料，未连接个人学籍系统或申请审批系统。
