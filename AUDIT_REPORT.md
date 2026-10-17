# HKMA FinSight 独立审计（2026-10-18）

> 历史审计快照：后续已逐题复核全部 19 道负题，当前标签与证据见 [RETRIEVAL_VALIDATION.md](RETRIEVAL_VALIDATION.md)，本地检查点见 [CHECKPOINT_REPORT.md](CHECKPOINT_REPORT.md)。本报告中的“其他负例尚未逐项核验”“所有变更未提交”描述的是前轮结束时状态。

结论：离线基线可复现；本次修复了可确定复现的引用、拒答及配置缺陷，但**系统仍不具备可靠的语义事实核验能力，不能宣称 production-ready**。没有付费 API 调用、模型下载、额外年报导入、索引重建、commit 或 push。

## 仓库状态与范围

- 实际目录：`/Users/chengjiajin/DoubaoWork/chats/2026-10-18/new-chat/financial-rag`。
- 分支 `main`，HEAD `78a1d76`，提交标题 `Merge pull request #42 from dasosis/feature/nightly-eval`。本地可见历史只有一个提交；未 fetch，不代表远端完整历史。
- 开始时已有 13 个已跟踪文件修改及 8 个未跟踪条目（其中一个是 validation 目录）。均保留，没有 reset、stash 或覆盖整份工作区。
- 本次生产代码仅修改 `financial_rag/rag.py`、`config.py`、`llm.py`；更新 `tests/test_rag.py` 并增加 `tests/test_audit_regressions.py`。补充 README、两份历史报告的更正提示，以及本报告和 validation 审计产物。
- 结束时仍在 `main`，所有变更未提交。准确状态保存在 `validation/audit_git_status.txt`；其中包括用户原有改动，不能将全部 diff 归为本次工作。

已检查：Git 状态/历史/差异；CODEX_HANDOFF.md、RETRIEVAL_VALIDATION.md；README 的架构、配置、HKMA、测试、评估部分；LICENSE、.gitignore、.env.example、pyproject.toml、requirements.txt、requirements-dev.txt；ingest.py、hkma.py、rag.py、hybrid.py、config.py、llm.py、embeddings.py；financials.py、router.py、followup.py、evaluate.py、eval_answers.py、api.py、start.py 的相关调用路径；conftest、rag、hkma、llm_embeddings、config、hybrid、upload、streaming、tracing、live 测试；两份原始 CSV、本地 PDF 和 Chroma SQLite。未读取或输出真实密钥。

## 已证实的问题与修复

| 严重度 | 证据 | 本次处理 |
| --- | --- | --- |
| P0 | 整数人数/金额、整数百分比不检查；问题中的数字可直接充当证据；完全无引用的答案可通过 | 检查整数及百分比，取消问题数字豁免；无引用、空答案、引用后无引用正文会失败。保留严格的 `not found in context` 拒答 |
| P0 | `page_label` 被当成 PDF 页码别名，可能接受错误的阅读器跳转页 | 只接受索引 `page + 1`；已有 HKMA 文件的 page_label 恰与阅读器页一致，该缺陷是对有偏移标签的 PDF 的确定性回归 |
| P0 | 不同 owner 的同名文件可合并数字，无法区分文档身份 | 引用涉及多 owner 同名文档时拒绝认定为有效证据 |
| P0 | 二次核验失败仍返回原始错误答案；流式输出先发布未经检查的草稿 | 二次失败替换为固定拒答，不返回错误数字；缓冲生成，检查后才发送 token。流式界面仍有步骤进度，但不会逐字展示未审草稿 |
| P0 | 默认 Ark 会回退取 OPENAI_API_KEY / OPENAI_BASE_URL，存在跨服务误发凭据风险 | Ark 只读取 ARK_API_KEY / ARK_BASE_URL，OpenAI 设置独立 |
| P1 | config 在调用模块执行 load_dotenv 之前就固定配置，直接导入/CLI 可能忽略 .env | config 自行先加载 .env；子进程回归验证，环境变量仍优先 |
| P1 | 生成工厂未显式限制超时，继承 SDK 长等待默认值 | 默认 30 秒超时、一次 SDK 重试；既有调用者覆盖仍生效 |
| P1 | N14 被标为“年报没有行政总裁薪酬证据” | 真实 PDF p.201（印刷 p.199）Table 5 有对应薪酬；保留原始 CSV，在历史报告/README 添加更正 |

最初新增的 10 个回归用例在修复前 **10 failed**，修复后通过。另增加正例、无证据不调用模型、流式不泄漏草稿、同名文件歧义，以及 HTTP 模拟测试。

## 原始证据与评估复核

- PDF：43,915,243 字节，334 页；SHA-256 `acc3cb3eb6d5289815355e84f7e27fec85bbcbaa7542e7a300f378967c367939`。
- SQLite 只读检查：一个集合，343 个 embedding / document；source_file 均为 AR2024_E.pdf，owner 均为 public，company 均为 Hong Kong Monetary Authority。
- **343/343** 分块去除空白后均为 metadata.page 指定的原始 PDF 页文本子串。检查未重写索引，也不证明整个表格解析保留了列关系。
- p.71 的 Eddie、p.184 的 218.8、p.211 的 Income and Expenditure Account、p.216 的 Changes in Equity 均实际存在。使用 Poppler 渲染并目视核对 p.184 和 p.201，确认阅读器页/印刷页差 2。
- p.201 Table 5 的 Chief Executive 列列出固定薪酬 7,361、浮动薪酬 2,172、其他福利 980，单位 HK$'000。表内是 annualised pay，脚注说明非全年任职时年化，并说明福利范围。不能擅自把分项相加当作原文明示总薪酬。
- 原 CSV 重算：英文端到端命中 13/15，中文 0/10；多语模型中文语义 Recall@5 从 2/10 到 5/10，但英文从 11/15 降至 8/15。这些是**既有 CSV 的算术复核，不是重新运行模型的独立实验**。
- 原标签下确有 9/19 个负题通过检索。但 N14 标注错误；其他负例没有逐项完成新的全文及表格人工标注，所以**不能报告校正后的拒答准确率**。检索放行也不等于生成模型真的给出错误答案。
- validation 原来只有两个 CSV，没有完整实验运行脚本、18 条逐字证据标注或模型 revision/依赖锁文件。因此历史实验无法仅靠这些产物逐步精确复现。本次不补造历史实验。
- 嵌入模型、cross-encoder、RERANK_SCORE_FLOOR=-2.0 均未改变。精确数字命中仍可绕过该粗相关性门限；这是检索偏好，绝非可回答性的保证。

只读证据复核可运行：

```bash
.venv/bin/python validation/audit_local_evidence.py
```

输出记录见 `validation/audit_local_evidence.json`。该脚本依赖现有 PDF/SQLite，不下载任何东西。

## 实际执行的测试

```bash
env RUN_LIVE_TESTS=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OPENAI_API_KEY= ARK_API_KEY= \
  .venv/bin/python -m pytest -p no:cacheprovider
.venv/bin/ruff check financial_rag/ tests/ start.py validation/audit_local_evidence.py
git diff --check
```

- 修改前：**496 passed / 62 skipped**，Ruff 通过。历史交接文件的 495 是较早快照，后来新增阈值测试后才是 496。
- 修改后：**516 passed / 62 skipped**，Ruff 与 diff whitespace 检查通过。日志见 `validation/audit_tests.txt`、`audit_ruff.txt`。
- 跳过：followup_live 5 个、planner_live 57 个；由 RUN_LIVE_TESTS 显式门禁控制。旧 skip 文案写 OpenAI，实际服务取决于配置；本次未开启。
- HTTP MockTransport 实际经过 ChatOpenAI 请求构造/SDK解析：验证 URL、model、messages、Bearer header、成功响应；401、429、500、超时和空 choices 均不伪装成功。max_retries=0 的测试确认仅一次模拟请求。
- API 层现有 RuntimeError → 503 路径与缺 key 工厂异常兼容；其他 SDK 错误仍走通用 500。未做真实 Ark HTTP 或部署冒烟验证。
- 没有重新跑原先多语言模型对比、真实 cross-encoder 全套基准或任何 live 测试。

## 状态与仍有的限制

**Verified（已验证）**：上述本地 PDF/索引一致性、页码/数字检查、确定性拒答和流式保护、配置分离/.env 加载、模拟 HTTP、完整离线测试、MIT 原文未修改。`.env` 当前不存在且不被跟踪；PDF、Chroma、venv 命中 .gitignore。已跟踪文件的常见长 sk-key / PEM 私钥模式扫描无候选；这不是全历史/所有密钥格式的安全证明。唯一 .env 类已跟踪文件是模板。

**Implemented but not fully tested（已实现但未完整验证）**：提示词要求具体事实/实体/时间/单位支持、无据拒答；配置化 Ark 客户端。提示词遵循性和真实模型拒答率未验证。

**Not implemented（未实现）**：可靠的 claim→evidence 语义蕴含验证。独立复现：拿真实 p.184 的投资收入段落，错误回答 `CEO salary was HK$218.8 billion (AR2024_E.pdf, p.184).`，`check_answer` 仍返回空问题列表。数字真实、地址真实，关系仍错。单位、正负号、年份与表列归属也不是本检查器能可靠判断的；不能将“通过检查”描述为事实已证实。检索 source 卡片是候选片段（前 300 字的焦点文本），不是逐条 claim 的完整证明；同页只展示首个卡片也可能省略后续块。

**Blocked / not exercised（未执行）**：当前环境无 ARK_API_KEY / OPENAI_API_KEY，未运行真实豆包生成；本任务本身也禁止付费调用，即便获得 key 也没有授权本次运行。未增加新模型、新报告、HKMA 统计 API 或中文检索功能。

其他复现注意：`start.py --reingest` 是通用 SEC 导入路径，会清库且不保留 HKMA 导入入口的 title/extract_figures=False 选项；本次未运行。将来切换模型应在备份/隔离索引中走 HKMA 专用导入，不应把通用重建命令当作经过验证的 HKMA 无损迁移流程。

## 后续真实 Ark 集成测试：精确且限额的步骤（本次未执行）

1. 单独确认愿意进行付费测试；在 Ark 控制台核对账号可用的模型/接入点、地区 endpoint 和费用限制。默认模型名的实际可用性未验证。
2. 在本机编辑已忽略的 `.env`，设置 `LLM_PROVIDER=doubao`、`ARK_API_KEY`、`ARK_BASE_URL`、`CHAT_MODEL`；不在聊天、日志或 Git 中粘贴密钥。可 `chmod 600 .env`。保持 EMBEDDING_PROVIDER=local、RERANKER=cross-encoder、RUN_LIVE_TESTS=0，不重建库。
3. 先运行上述离线测试。然后从项目目录运行下面的**独立集成检查**：固定真实证据页，共三次模型请求，不自动重试；任何错误立即退出。这里直接验证生成与响应解析，不调用规划/检索/LLM rerank，避免额外费用。

```python
# 保存为本地临时脚本或在 python stdin 执行；用仓库 .venv/bin/python。
from pypdf import PdfReader
from langchain_core.documents import Document
from financial_rag import rag
from financial_rag.llm import get_chat_llm

reader = PdfReader('data/AR2024_E.pdf')
model = get_chat_llm(timeout=30, max_retries=0, max_tokens=512)
cases = [
    (184, 'What was the Exchange Fund investment income in 2024?'),
    (201, 'What were the Chief Executive fixed pay and variable pay in 2024?'),
    (184, 'What was the return on Japanese equities specifically in Q3 2024?'),
]
for page, question in cases:
    doc = Document(page_content=reader.pages[page - 1].extract_text(),
                   metadata={'source_file': 'AR2024_E.pdf', 'page': page - 1})
    response = model.invoke([
        ('system', rag.SYSTEM_PROMPT.format(context=rag._format_docs([doc]))),
        ('human', question),
    ])
    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError('Expected non-empty text response')
    print(question, response.content, rag.check_answer(response.content, question, [doc]))
```

4. 人工对照 PDF：第一题应引用 p.184 的投资收入；第二题区分 p.201 的固定/浮动薪酬与 HK$'000 单位；第三题应说明给定片段无日本股票第三季度分项，不能用全年股票收益替代。空问题列表不等于正确，必须审查关系、时期、单位和引用。
5. 保存非敏感的模型 ID、时间、请求数量、usage、答案及人工判定；不要保存 header/key。若 API 不支持请求参数，先查账号对应接口要求再改，不自动尝试多种付费调用。此检查通过后，再另行授权小规模全链路正负题测试；不要直接开启全部 62 个 live 用例。
