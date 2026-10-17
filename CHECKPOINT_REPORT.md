# HKMA FinSight 本地检查点记录

日期：2026-10-18。按用户授权将已有改造、上一轮审计修复及本轮标签更正保存到本地 Git；不推送。此次没有开发产品功能、修改模型或阈值、调用付费 API、下载模型或重建索引。

## 起点与工作保留

起点为 main / `78a1d76119a4551cde536207eef32ab69275c93a`。检查了 status、HEAD、diff、差异统计和 whitespace。原有修改均保留：文件指纹复核显示本轮仅修改原有的 README、AUDIT_REPORT、CODEX_HANDOFF、RETRIEVAL_VALIDATION、.gitignore 和 .gitattributes；没有删除原文件，也没有改变上一轮已经通过测试的生产代码或测试逻辑。暂存检查发现原有未跟踪 tests/test_hkma.py 的末尾多余空行，已仅移除该空行。旧的 RETRIEVAL_VALIDATION 完整保存在 `validation/history/RETRIEVAL_VALIDATION.before_label_review.md`。

## 标签和实际证据

逐一核验原先 19 道负题，结果为：充分证据 1、部分证据 3、证据不足 14、问题含糊 1。

- N14 改为充分证据：PDF p.201 Table 5，2024 年化、HK$千；行政总裁固定薪酬 7,361、浮动薪酬 2,172、其他福利 980。分别回答，不混用不同薪酬口径或擅造合计。
- N16 改为需澄清：PDF p.199 的 2024 编制 1,112；p.200 Table 4 为 2025-01-01，实际在岗分项 15 和 1,037，不能与编制 16 / 1,117 混淆。合计需要计算，非原文直接印出的数值；原问题需明确日期及口径。
- N12 日本股票 Q3 收益、N13 指定日期汇率、N15 网安支出标为部分证据。分别核对 p.183–185（全年指数/基金数据）、p.14/83–84（年底报价/区间/曲线）、p.101/195–197（措施及大类开支），确认不能据这些内容回答所问细分数字。
- N01–N11、N17–N19 保留证据不足。检查了美联储/欧洲央行/比特币/JPMorgan/沪港通/日本央行及离题问题的主体与同义词；上下文命中和不相干表格均标为 context_only。没有支持页时明确留空，不制造“证明不存在”的引用。

完整原题、每道题的页码、摘录、搜索词/命中页、判断及预期行为见 `validation/negative_cases_reviewed.json` 和 `RETRIEVAL_VALIDATION.md`。搜索覆盖全部 334 页，目视复核关键图表。缺失判断仅限当前报告和本轮审阅，不是绝对不存在的证明。

两个原始 CSV 字节不变；其 SHA-256 已写入证据台账并经回归测试锁定。旧 9/19、E 类 5/5 不能作为可靠漏拒率，保留在明确标记的历史材料中。本轮没有运行模型，不能把文档级标签当成旧 Top-5 答案质量，也没有声称测得新准确率。

## 实际检查结果

- `validation/verify_negative_case_evidence.py`：19 题、334 页、28 条摘录、76 组搜索记录全部通过；原始 CSV 指纹不变。
- `RUN_LIVE_TESTS=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_API_KEY= ARK_API_KEY= .venv/bin/python -m pytest -p no:cacheprovider`：**520 passed, 62 skipped**，耗时详见最终日志。62 个 live 用例保持跳过。
- `.venv/bin/ruff check financial_rag/ tests/ start.py validation/*.py`：All checks passed。
- `git diff --check`：通过；提交前另检查暂存差异。
- 结果日志：`validation/negative_review_verification.json`、`checkpoint_pytest.txt`、`checkpoint_ruff.txt`。

新增四个离线数据回归测试：19 题不重不漏且题干不变、原始 CSV 不变、上下文页不可混作答案支持、薪酬和人数更正标签受保护。另有只读 PDF 校验脚本验证真实原文，测试未假造报告摘录。

## 提交安全与范围

提交包括原有 HKMA 导入/模型配置、本轮之前已经完成的引用与拒答修复、相关测试、文档和评估数据。本轮新增的实现仅为证据校验脚本与数据回归测试。

`.env` 当前不存在；`.env`、`.env.production`、本地 PDF、Chroma SQLite 和虚拟环境均通过 git check-ignore。增加 .gitattributes 对两份原始 CSV 设置 -text 并声明 cr-at-eol，保留原始 CRLF 且继续检查其他空白错误，防止 Git 换行转换破坏历史指纹。增加 `.env.*` 忽略规则并保留模板例外 `!.env.example`。提交候选文件中没有 PDF、数据库、虚拟环境或非模板 .env。常见长 API key、AWS access key 和 PEM/OpenSSH 私钥模式扫描没有候选；模板/测试中的 dummy 值不是凭据。这不等价于所有历史或所有密钥格式的安全保证。

MIT LICENSE 与原 HEAD 逐字节相同，版权归属保留。最终提交清单可用 `git show --stat HEAD` 查看，commit hash 用 `git rev-parse HEAD` 获取（不把提交自身的 hash 写入自身文件）。

## 尚未解决

语义蕴含核验仍未实现，真实数字搭配错误主体/单位仍可能通过；中文端到端检索不足；未验证真实豆包生成及拒答遵循性；原始实验缺少完整 Top-5 原文和运行脚本，无法还原生成准确率。N16 等问题需要明确时间/口径。上述限制未在本轮扩大开发范围处理。
