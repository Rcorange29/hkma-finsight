# HKMA FinSight — Codex 独立审查交接文档（CODEX_HANDOFF）

> 最新标签与检查点：见 [RETRIEVAL_VALIDATION.md](RETRIEVAL_VALIDATION.md) 和 [CHECKPOINT_REPORT.md](CHECKPOINT_REPORT.md)。下文为历史交接记录，不代表当前标签或 Git 状态。

> 后续独立审计更正：见 [AUDIT_REPORT.md](AUDIT_REPORT.md)。本文保留为历史记录；当前代码已取消 Ark 对 OpenAI 密钥/URL 的回退、改为检查失败拒答，并收紧引用检查。原先“证据不足”结论不代表语义核验通过。

本文件供独立代码审查使用。它只记录**实际执行并验证过**的事实；未完成、未验证的事项在第 6、7 节明确标注，不做任何掩饰。

- 基础仓库（上游）：`https://github.com/dasosis/financial-rag`（SEC 10-K 财报 RAG，LangChain + ChromaDB + OpenAI）
- 本地路径：`/Users/chengjiajin/DoubaoWork/chats/2026-10-18/new-chat/financial-rag`
- 分支：`main`（基线提交 `78a1d76`，所有改动均未 commit，停留在工作区）
- 许可证：**MIT，Copyright (c) 2026 Soumya Snigdha Das**，未修改；本次改动不删除、不替换上游版权与许可。

---

## 1. 项目当前架构

整体仍是上游的 RAG 架构，本次只做最小改造，未更换技术栈：

```
PDF（HKMA 年报）
  └─ ingest.load_pdf（pypdf，一页一个 Document，metadata["page"] 为 0 基）
       └─ ingest.ingest_file（表格感知分块 CHUNK_SIZE=1024/OVERLAP=128）
            └─ embeddings.get_embeddings()  ── 默认本地 BGE（BAAI/bge-small-en-v1.5, 384 维，离线）
            └─ ChromaDB（集合 financial_reports，目录 chroma_db/）

问答（rag.query）
  ├─ retrieve_and_rerank
  │    ├─ 语义检索（similarity_search_with_relevance_scores，门限 SCORE_THRESHOLD）
  │    ├─ BM25 关键词检索（hybrid.py，停用词过滤）
  │    ├─ RRF 融合
  │    ├─ cross-encoder 重排（cross-encoder/ms-marco-MiniLM-L-6-v2）
  │    ├─ 【新增】cross-encoder 门限 RERANK_SCORE_FLOOR（精确数字匹配豁免）
  │    └─ 可选 LLM 重排（有 key 时默认 llm，无 key 时本地 cross-encoder）
  ├─ 无证据 → 固定拒答语，返回 sources=[]，不调用模型
  └─ answer_checked → llm.get_chat_llm()（OpenAI 兼容；默认豆包/火山方舟）
       └─ check_answer：答案中的每个数字必须出现在所引页/检索片段；引用页必须真实检索到；
                       失败重试一次，仍失败则追加 "Not verified" 警告
```

服务入口（均为上游已有）：
- `start.py`：本地启动 FastAPI 后端（`financial_rag/api.py`）+ Streamlit UI（`financial_rag/app.py`），支持 `--reingest`。
- CLI：`python -m financial_rag.rag "<question>"`（问答）、`python -m financial_rag.hkma ...`（年报导入）。

新增的三个"工厂"模块把模型构造集中到一处，其余模块只调用工厂：
- `llm.py` → `get_chat_llm()`；`embeddings.py` → `get_embeddings()`。
- 全仓已无绕过工厂直接 `ChatOpenAI(...)` / `OpenAIEmbeddings(...)` 的调用（grep 验证）。

---

## 2. 已实现且实际验证的功能

以下均在真实数据/真实命令上跑过，非 mock（mock 仅用于离线单元测试）。

| 能力 | 状态 | 实测证据 |
| --- | --- | --- |
| HKMA 年报在线发现 | 已验证 | 解析 2023/2024/2025 年份页，正确选出合并版 `AR2023_E.pdf` / `AR2024_E.pdf` / `AR_2025E.pdf`（23 个章节文件之外的唯一合并 PDF） |
| 年报下载与入库 | 已验证 | `data/AR2024_E.pdf`，43,915,243 字节（约 42MB），**334 页**；ChromaDB **343 个分块**，`company="Hong Kong Monetary Authority"`，SEC 三表引擎关闭（`figures=0`） |
| PDF 页码保留 | 已验证 | 用合并版单一连续 PDF；0 基索引 → 阅读器 1 基页码 `page_number(idx)=idx+1` |
| 可核验原文引用 | 已验证 | pypdf 逐页复核：PDF **p.184** 含 `218.8`（投资收入）；**p.71** 含 `Eddie`（行政总裁）；**p.211** 含 `Income and Expenditure Account`；**p.216** 含 `Changes in Equity` |
| 本地嵌入 | 已验证 | BGE 加载成功，384 维；入库/检索/重排全程离线、零付费 |
| 本地重排 | 已验证 | cross-encoder ms-marco-MiniLM-L-6-v2 正常，分数已记录在 metadata |
| 证据不足拒答 | 已验证并加固 | 11 个离题问题（天气、电影、菜谱、笑话等）全部返回固定拒答语、`sources=[]`、不调模型；见第 7 节验收修复 |
| 有证据无 key 清晰报错 | 已验证 | CLI 打印检索证据页 + 配置提示并 `exit 1`；HTTP 返回 **503**，detail 指引设置 `ARK_API_KEY`；不会输出 traceback 或伪造答案 |
| 豆包 API 配置 | 已实现（离线） | 配置解析、工厂、`.env.example`、mock 测试均就绪；**真实调用因无 key 未执行**（见第 6 节） |
| FastAPI | 已验证 | `/health`→`ok`；`/documents` 列出 343 块；`/query` 离题 200 拒答、正题无 key 503 |

**页内偏移说明**：2024 合并版有 2 页前置页，因此**印刷页码 = PDF 页码 − 2**（如 PDF p.184 显示印刷页 182）。引用一律使用 PDF 页码，因为它能直接在阅读器中跳转。

---

## 3. 已修改/新增的文件及其作用

### 新增（5 个，git 未跟踪）
| 文件 | 行数 | 作用 |
| --- | --- | --- |
| `financial_rag/hkma.py` | 240 | HKMA 年报发现/下载/入库 + CLI（`--year/--max-pages/--download-only/--force`）；HTTP 薄封装可被测试替换；只调 `ingest.ingest_file`，不重复实现 |
| `financial_rag/llm.py` | 52 | `LLMConfigError`、`require_api_key()`、`get_chat_llm()`；在调用时查找 `langchain_openai.ChatOpenAI` 以保证测试 monkeypatch 生效 |
| `financial_rag/embeddings.py` | 37 | `get_embeddings()`：默认本地 `HuggingFaceEmbeddings(bge-small-en, normalize_embeddings=True)`，可切 OpenAI |
| `tests/test_hkma.py` | 240 | 12 个离线测试：发现/URL 归一化/异常分支/下载缓存/非 PDF 拒绝/标题与跳过 SEC 数字/页码引用/无证据拒答/CE 门限 |
| `tests/test_llm_embeddings.py` | 98 | 6 个离线测试：豆包 base_url/model/key 透传、OpenAI 无自定义 base_url、缺 key 报错、嵌入默认本地/可切 OpenAI、本地阈值 |

### 修改（13 个，git diff 可见）
| 文件 | 作用 |
| --- | --- |
| `financial_rag/config.py` | 重写：`LLM_PROVIDER`（默认 doubao）、`CHAT_MODEL`、`LLM_BASE_URL`（ARK→OPENAI→默认）、`LLM_API_KEY`（ARK→OPENAI）、`EMBEDDING_PROVIDER`（默认 local）、`LOCAL_EMBEDDING_MODEL`；保留 `EMBEDDING_MODEL`/`RERANKER_MODEL` |
| `financial_rag/rag.py` | 模型改走工厂；提示词由 SEC 10-K 通用化为年报；`SCORE_THRESHOLD` provider 感知（本地 0.35/OpenAI 0.10）；**新增 `RERANK_SCORE_FLOOR` 门限**；无 key 时默认本地 cross-encoder；CLI 捕获 `LLMConfigError` 打印证据 |
| `financial_rag/hybrid.py` | **新增 `QUESTION_FILLER` 停用词**（give/tell/please 等对话动词），保留金融同形词（state/share/return 等） |
| `financial_rag/ingest.py` | 向量库用 `get_embeddings`；`ingest_file` 新增 `title=None`、`extract_figures=True`（HKMA 传 False 并清空旧 facts） |
| `financial_rag/router.py` / `followup.py` / `financials.py` / `evaluate.py` | 统一改走 `get_chat_llm` / `get_embeddings`；followup 把模型构造移入 try，无 key 时优雅兜底 |
| `start.py` | `check_env()` 放宽：无 key 仅警告（入库/检索本地可用，仅问答需 key），同时解析 `.env` 与已导出环境变量 |
| `tests/test_upload.py` | 两处 patch 目标改为新工厂 |
| `tests/test_hybrid.py` | 新增 2 个测试：对话停用词不产生关键词命中、金融同形词不被停用 |
| `.env.example` | 重写为 Ark/本地嵌入/阈值说明，保留原 SEC/Cognito/OTEL 注释；**仅占位符，无真实密钥** |
| `README.md` | 新增「HKMA Annual Reports (FinSight)」「Models: Doubao and local embeddings」两节；更新模块表/Quickstart/配置表/测试说明；保留全部 SEC 内容与 MIT |

> `requirements.txt` **最终无改动**（验收中移除了一度加入的 `fonttools`，见第 7 节）。

---

## 4. 运行项目的准确命令

```bash
cd /Users/chengjiajin/DoubaoWork/chats/2026-10-18/new-chat/financial-rag
source .venv/bin/activate            # Python 3.12.15（venv 已在仓库内）

# 下载并入库 HKMA 2024 年报（本地 BGE，首次会下载约 130MB 嵌入 + 90MB 重排模型）
python -m financial_rag.hkma --year 2024
#   仅下载：python -m financial_rag.hkma --year 2024 --download-only
#   仅前 N 页：python -m financial_rag.hkma --year 2024 --max-pages 40
#   重新下载：python -m financial_rag.hkma --year 2024 --force

# 命令行问答（无 key 时打印检索证据并提示配置；有 key 时给最终答案）
python -m financial_rag.rag "What was the Exchange Fund investment income in 2024?"

# 启动 FastAPI + Streamlit（重建向量库用 --reingest）
python start.py
# 仅后端冒烟：
AUTH_DISABLED=1 python -m uvicorn financial_rag.api:app --host 127.0.0.1 --port 8012
#   本地模式请求头：X-User-Id: local
```

注意：重建/切换嵌入模型用 `python start.py --reingest`（`python -m financial_rag.ingest` 只支持 `--fetch-samples`，没有 `--reingest`，验收中已据此更正文档）。

---

## 5. 测试命令与实际测试结果

```bash
source .venv/bin/activate
python -m pytest -p no:cacheprovider
ruff check financial_rag/ tests/ start.py
```

**实际结果（本次验收最终值）：`495 passed, 62 skipped`，`ruff` All checks passed。**
- 上游基线：475 passed / 62 skipped。
- 本次新增 **20 个离线测试**（test_hkma 12、test_llm_embeddings 6、test_hybrid 2），全部通过。
- 所有新测试均使用 `DeterministicFakeEmbedding`、假 `_FakeReranker` 与注入的 `fetch_text/fetch_bytes`，不联网、不下载模型、不产生付费调用。

**62 个跳过的具体内容与原因（全部为上游既有、默认跳过的 live 测试）：**
- `tests/test_followup_live.py`：5 个参数化用例。
- `tests/test_planner_live.py`：57 个参数化用例（5 组）。
- 二者统一门禁：`pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="calls OpenAI; set RUN_LIVE_TESTS=1")`。
- 跳过原因：它们调用真实聊天模型（联网、付费），需显式 `RUN_LIVE_TESTS=1` 且配置 key 才运行。基线本来就跳过这 62 个，验收未运行它们。
- 小提示：这两个文件的跳过文案仍写 "calls OpenAI"，改造后实际会调用"所配置的聊天 provider"（默认豆包）；行为正确，仅文案略陈旧，属上游文件，未改。

---

## 6. 尚未完成的功能（明确不冒充完成）

1. **真实豆包生成答案未执行**：当前环境**没有 `ARK_API_KEY`**（仓库内也没有 `.env`）。因此"最终生成答案"这一步从未真实调用，**没有也不会伪造豆包回答**。配置代码、错误提示与 mock 测试已就绪；拿到 key 后即可直接运行（见第 8 节），`check_answer` 引用核验/重试/警告链路会自动生效。
2. **2023 / 2025 等其他年份未入库**：发现逻辑已对三年在线验证，但只实际下载/入库了 2024（用户明确要求不导入更多年份）。
3. **live 测试未运行**：`RUN_LIVE_TESTS=1` 的 62 个用例需要真实模型与 key，未执行。
4. **SEC 三表确定性引擎对 HKMA 不适用**：HKMA 问答是"文本检索 + 页码引用"，不会计算结构化财务数字（这是设计取舍，不是缺陷）。
5. 评估脚本（RAGAs/`evaluate.py`）未在 HKMA 语料上跑过。

---

## 7. 已知问题、验收修复、需重点审查的位置

### 7.1 验收中发现并已修复的问题（均有明确证据，非"为重构而重构"）
1. **离题问题因常见动词绕过拒答门限（已修复）**。证据：`"give me a recipe for chocolate cake"` 里，`hybrid.py` 的关键词分词未把 `give` 当停用词，而 `give` 在年报中出现 12 次 → 12 个 BM25 命中，绕过语义门限，把 cross-encoder 分数仅 **−11.04**（完全无关）的分块送入答案环节。修复：在 `hybrid.py` 增加 `QUESTION_FILLER`（give/tell/please/show/find/… 等对话/礼貌动词），同时刻意保留 `state/share/return/rate/report/use/list` 等金融同形词；新增 2 个纯函数测试。
2. **cross-encoder 分数未被用作拒答门限（已修复）**。证据：语义门限（0.35）与关键词命中之外，重排后没有再做相关性过滤，导致 `weather today`（语义 0.388 + 关键词 "weather/today"）、`recommend a movie`（关键词 "recommend"）等以 −8.9～−10.9 的重排分进入答案。修复：在 `rag.py` 重排后增加 `RERANK_SCORE_FLOOR`（默认 **−2.0**，原始 logit，env 可覆盖）；**引用了页面真实数字的精确匹配（`exact[cid]`）豁免**；全部低于门限则返回空 → 拒答。
   - 校准（真实库，仅本地 cross-encoder，无付费）：14 个正题最佳 CE 分区间 **+0.55～+9.17**；11 个离题问题修复前最佳 **−8.98～−11.04**（修复后全部返回 0 文档）。门限 −2.0 两侧均有较大余量。
   - 修复后 495 全绿；假 cross-encoder 返回 0.0 ≥ −2.0，故既有离线测试不受影响。
3. **移除不必要依赖 `fonttools`**。phase 1 曾以为它改善 CFF/Type1 字体提取并加入 requirements；验收实测：屏蔽 `fontTools` 后，对 p.71/184/211/216（含密集财务报表页）pypdf 提取结果**逐字节相同**，且 pypdf 不声明该依赖（它仅被 matplotlib 间接安装）。已从 requirements 移除，requirements.txt 恢复为与上游一致。
4. **更正不准确的重建命令**：`config.py`/`embeddings.py` 误写为 `python -m financial_rag.ingest --reingest`（该子命令不存在），已更正为真实存在的 `python start.py --reingest`。
5. **`hkma.py` User-Agent 文案**：原来引用了它并不读取的 `SEC_USER_AGENT`；已改为 `os.getenv("SEC_USER_AGENT", "financial-rag/hkma-annual-report (research)")`，行为诚实。

### 7.2 建议 Codex 重点审查的位置
- `financial_rag/rag.py`：
  - `SCORE_THRESHOLD`（58–60 行附近）与 `RERANK_SCORE_FLOOR`（约 70 行）的取值是否合理、env 覆盖是否正确。
  - `retrieve_and_rerank` 中"精确数字匹配豁免门限"的逻辑（`exact[cid]` 与 `hybrid.has_amounts`），确认它不会成为绕过拒答的新口子（例如问题含很多数字时）。
  - 两个弱正题的余量较小：`foreign currency reserves`(+0.58)、`authorized institutions`(+0.55)。若上调门限到 0 以上可能误拒；建议保留 −2.0 或在更多真实问题上再校准。
- `financial_rag/hybrid.py`：`QUESTION_FILLER` 词表是否覆盖足够多的对话动词、是否误伤任何金融术语（已用测试锁定 state/share/return 等不被停用）。
- `financial_rag/llm.py`：`base_url`/`api_key` 对豆包与 OpenAI 的分流是否正确；`LLMConfigError` 是 `RuntimeError` 子类，API 层 `except RuntimeError` 能转 503，建议复核。
- `financial_rag/hkma.py`：`discover_annual_report` 对"唯一非章节 PDF"的假设；HKMA 网站结构若变化，会抛 `HKMAError` 而非静默选错文件。

### 7.3 其它已知边界（未改，供知悉）
- 嵌入模型是 `bge-small-**en**`（英文）。HKMA 年报为英文，正常；若未来要问中文/中英混排内容，应换多语嵌入模型并重建库。
- `langchain_community.embeddings.HuggingFaceEmbeddings` 当前可用但有 DeprecationWarning；官方新包是 `langchain-huggingface`。功能正常，未更换以免无谓动栈（可作为后续清理项）。
- BGE 384 维与 OpenAI 1536 维不兼容；切换 `EMBEDDING_PROVIDER` 必须 `--reingest` 重建（已在 README/.env 写明）。
- 首次下载模型依赖 huggingface.co 可达；网络不通可设 `HF_ENDPOINT=https://hf-mirror.com`。

---

## 8. 豆包（火山方舟 Ark）配置方法（不含任何真实密钥）

1. 申请 API Key（网页）：`https://console.volcengine.com/ark/region:ark+cn-beijing/apikey`
2. 在仓库根创建 `.env`（已被 `.gitignore` 忽略，不会提交）：
   ```dotenv
   ARK_API_KEY=你的密钥                      # 仅本地填写，勿提交
   LLM_PROVIDER=doubao                        # 默认值，可省略
   CHAT_MODEL=doubao-seed-2-1-pro-260628      # 也可填 ep-xxxx 接入点 ID
   ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3   # 默认值，可省略
   ```
3. 也可改用 OpenAI 或其他 OpenAI 兼容服务：
   ```dotenv
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   OPENAI_BASE_URL=https://...   # 可选，其他兼容网关
   CHAT_MODEL=gpt-4o-mini
   ```
4. 配置优先级：provider=doubao 时 base_url 取 `ARK_BASE_URL`→`OPENAI_BASE_URL`→Ark 默认；key 取 `ARK_API_KEY`→`OPENAI_API_KEY`。
5. 无 key 时：入库/检索/重排/拒答全部本地可用；只有"最终答案/改写/规划"会抛 `LLMConfigError`（CLI 打印证据并退出码 1，HTTP 503）。

密钥安全核查：仓库无 `.env`；`.env.example` 仅含占位符；已扫描源码，未发现硬编码密钥；`data/`、`chroma_db/`、`.env`、`.venv/`、`__pycache__/` 均被 gitignore。

---

## 9. 原开源项目与新增代码的边界

- **上游（保留不改其设计）**：LangChain + ChromaDB 的 RAG 主体、混合检索（语义 + BM25 + RRF）、cross-encoder 重排、答案核验 `check_answer`/重试/警告、FastAPI + Streamlit、SEC EDGAR/XBRL 三表引擎、用量限制、Cognito 登录、OTEL 追踪、Docker/compose、以及绝大部分 SEC 测试。
- **新增、自包含、可整块识别**：`hkma.py`（HKMA 数据接入）、`llm.py`（聊天模型工厂）、`embeddings.py`（嵌入工厂）、`tests/test_hkma.py`、`tests/test_llm_embeddings.py`。
- **对上游文件的侵入式改动**：仅为"把模型构造改为调用工厂""把提示词从 SEC 通用化为年报""新增两处拒答门限与停用词""放宽无 key 启动检查"。这些改动用 `git diff` 可逐行审阅，且上游 475 个测试全部继续通过（含被改的 router/followup/financials/ingest/rag 等），说明 SEC 路径未被破坏。
- **许可证与署名**：保留根目录 `LICENSE`（MIT）与 README 中的上游仓库地址；未删除任何版权行（`git diff` 已确认无 Copyright/License 行被移除）。建议在分发时保留对原作者 Soumya Snigdha Das 的署名；HKMA 年报版权归香港金融管理局，引用答案均标注官方 PDF 文件名与页码。

---

## 附：一键复现本次验收（离线，不付费）

```bash
cd /Users/chengjiajin/DoubaoWork/chats/2026-10-18/new-chat/financial-rag
source .venv/bin/activate
python -m pytest -p no:cacheprovider        # 期望 495 passed, 62 skipped
ruff check financial_rag/ tests/ start.py   # 期望 All checks passed
python -m financial_rag.rag "What was the Exchange Fund investment income in 2024?"
#   → 无 key：打印 p.180/184/211/212 等证据 + 设置 ARK_API_KEY 提示，退出码 1
python -m financial_rag.rag "what is the weather today"
#   → 固定拒答语，sources=[]，退出码 0
```
