# HKMA FinSight 检索验证：更正后的证据标签

日期：2026-10-18。当前入口为本文件；旧实验描述完整保存在 [历史报告](validation/history/RETRIEVAL_VALIDATION.before_label_review.md)。

## 结论与适用范围

原先全部标成无据负例的 19 题已逐题复核：**充分 1、部分 3、不足 14、含糊 1**。这是对整份 HKMA 2024 年报的人工证据标签，不是对既有 Top-5 片段的答案质量评分，更不是本次运行模型得出的拒答准确率。

- N14：由无据负题改为充分证据，按 2024 年报口径分项作答。
- N16：由证据不足负题改为需澄清；年报有人员数据，但日期、编制/实际在岗与合并总数口径必须明确。
- N12、N13、N15：保留不能完整回答的判断，明确改为部分证据；不能把全年/全市场/开支大类冒充所问分项。
- N01–N11、N17–N19：复核后仍没有足够证据；部分存在同词或同机构的上下文，不能作为答案支持。

原始 `threshold_per_question.csv` 和 `retrieval_per_question.csv` 字节保持不变，只用作历史实验。旧的 **9/19 漏拒、E 类 5/5 漏拒** 已撤销为可靠准确率指标，不计算一个貌似“修正”的替代准确率。旧实验没有真实生成输出，检索放行不等于回答错误。

## 核验方法和文件

- 源文件 `data/AR2024_E.pdf`，334 页，SHA-256 `acc3cb3eb6d5289815355e84f7e27fec85bbcbaa7542e7a300f378967c367939`。
- 搜索全部 334 页提取文本，按每题主体、时间、指标及同义词定位；阅读相关完整页面。p.84、183、184、196、200、201 的图表/表格经渲染目视核对（184 在前轮审计中，其他本轮复核）。
- 记录在 [JSON 证据台账](validation/negative_cases_reviewed.json)、[CSV 标签表](validation/negative_cases_reviewed.csv)。JSON 包含每题原问题、分类、PDF 页码、真实摘录、证据角色、搜索式和命中页、判断理由及期望行为。
- `answer_support` 是可支持所问事实的原文；`partial_support` 只支持部分或相邻事实；`context_only` 是会误导检索的相关文字。无证据题的支持页为空，绝不编造“证明不存在”的页码。
- 页码一律为物理 PDF 1 基页码。p.201 对应印刷 199、p.200 对应印刷 198，但不可将“减 2”泛化到整份报告；例如附录 PDF p.322 的页内文字标为 321。
- “没有足够证据”意为本轮搜索与相关页审阅未找到答案；不是数学意义的不存在证明，也未穷尽图像内不可提取的文字。未查询外部网站或用常识补答案。

## 两个关键标签

**N14 行政总裁薪酬（充分）**：PDF p.201 Table 5，标题明确 2024，单位 HK$'000，年化口径。Chief Executive 列：Fixed pay 7,361；Variable pay 2,172；Other benefits 980。应分别报告，不能将福利当作固定工资，也不能将自己相加的总额说成原文总额。脚注 a 涉及非全年任职时年化处理；脚注 c 涵盖公积金/酬金、医疗、人寿保险和年假。这里以年报年度作为明确作答范围；如果要求其他年度或单一 salary 定义，再澄清。

**N16 员工总数（含糊）**：PDF p.199 说明 2024 establishment 为 1,112；p.200 Table 4 日期为 2025-01-01，四列依次是 Senior staff Establishment/Strength、Others Establishment/Strength，Total 行分别为 16 / 15 / 1,117 / 1,037。它提供实际人员分项，不能标作完全无人员证据。15+1,037=1,052 是人工加总，不是页面直接印出的全体人数；当前生成提示词禁止推算，不能把“未算出合计”当作“年报没有信息”。先问清日期与口径，或先明确提供带日期的分项。

## 逐题结论

| ID | 问题主题 | 当前分类 | 核验页与结论 |
|---|---|---|---|
| N01 | What is the size of the Federal Reserve's balance sheet (total assets) at the end of 2024? | 没有足够证据回答 | 75, 84, 85, 213, 214。美联储命中讨论降息；资产负债表属于外汇基金，并非美联储。未找到美联储 2024 年底总资产。 |
| N02 | What is the European Central Bank's main refinancing operations rate in 2024? | 没有足够证据回答 | 14, 41, 123, 180, 183。只有主要央行降息概述及香港 Base Rate；refinancing 命中房贷安排。未找到欧洲央行主要再融资利率；原题也未指定年内日期。 |
| N03 | How much bitcoin does the HKMA hold on its balance sheet? | 没有足够证据回答 | 106, 181, 213。未检出 Bitcoin/BTC；报告的数字资产监管和金融资产类别不能证明 HKMA 比特币持仓量，也不能推出为零。 |
| N04 | What was JPMorgan Chase's net income for 2024? | 没有足够证据回答 | 180, 213, 314。唯一直接机构命中为认可机构名单，非 JPMorgan Chase 集团利润表；未见 2024 净利润。 |
| N05 | What are the daily trading quotas for Shanghai-Hong Kong Stock Connect? | 没有足够证据回答 | 7, 106, 149, 191。quota 命中跨境理财通个人投资者额度，不是沪港通每日北向/南向交易额度；未检出 Stock Connect。 |
| N06 | What is the Bank of Japan's yield curve control target for 10-year JGBs? | 没有足够证据回答 | 9, 148, 153, 165, 269, 310, 322。日本央行命中支付结算系统说明；收益率曲线命中香港发债/估值，不支持日本十年国债 YCC 目标。原题未指定日期，亦不能用外部常识补答。 |
| N07 | What will the weather be like in Hong Kong tomorrow? | 没有足够证据回答 | 5, 84, 166, 226, 237。报告讨论恶劣天气交易机制，不提供相对提问时间“明天”的香港天气预测。 |
| N08 | Can you recommend a good movie to watch this weekend? | 没有足够证据回答 | 191。电影只用于理财通宣传视频说明，不支持周末电影推荐。 |
| N09 | Give me a recipe for spaghetti bolognese. | 没有足够证据回答 | 无支持页；全文搜索。全文未检出意面食谱；不存在可诚实引用为做法证据的页码。 |
| N10 | Who won the FIFA World Cup in 2022? | 没有足够证据回答 | 322。世界杯/足球查询无命中；旧检索命中 p.322 是银行按地区分类表，其国家名和年份不证明冠军。 |
| N11 | How do I bake a chocolate cake? | 没有足够证据回答 | 无支持页；全文搜索。未检出巧克力蛋糕制作证据；支持页为空，不能制造引用。 |
| N12 | What was the Exchange Fund's return on Japanese equities specifically in the third quarter of 2024? | 有部分证据，但不足以完整回答 | 180, 181, 183, 184, 185。有 TOPIX 全年市场指数涨幅及外汇基金全年股票收益；没有日本股票第三季度的基金收益分项。市场指数不是基金组合，全年不是 Q3。 |
| N13 | What was the HKD/USD exchange rate at market close on 15 March 2024? | 有部分证据，但不足以完整回答 | 14, 83, 84。有兑换保证区间和全年市场汇率曲线，但没有 2024-03-15 收市的精确报价。不能从低分辨率曲线读出精确收市价。 |
| N14 | What is the annual salary of the HKMA Chief Executive? | 有充分证据，可以回答 | 199, 200, 201。以本题集的 2024 年报为时间范围，Table 5 的 Chief Executive 列足以分项回答：固定薪酬 7,361、浮动薪酬 2,172、其他福利 980（HK$千，年化口径）。不能继续标成无据负题；salary 不应未经说明等同含福利总包。 |
| N15 | How much did the HKMA spend on cybersecurity in 2024? | 有部分证据，但不足以完整回答 | 6, 92, 101, 195, 196, 197。有自用网络安全措施和行政开支大类；Table 2 的 221（HK$百万，2024 Actual）是办公室及电脑设备维护，不是网络安全专用支出。未见该项独立金额。 |
| N16 | How many employees does the HKMA have in total? | 问题本身含糊，需要澄清 | 188, 190, 195, 199, 200, 201, 203。原题无日期/口径；2024 编制 1,112 不等于实际员工数。Table 4 是 2025-01-01，Senior staff Strength=15、Others Strength=1,037；不能把各列或培训人次当总人数。不存在“年报未披露人员证据”的结论。 |
| N17 | 明天香港天气怎么样？ | 没有足够证据回答 | 5, 166, 226, 237。与 N07 同义：只有恶劣天气交易机制，非明日香港天气预报。 |
| N18 | 推荐一部好看的电影吧。 | 没有足够证据回答 | 191。与 N08 同类：宣传视频使用电影片段不构成电影推荐依据。 |
| N19 | 怎么做红烧肉？ | 没有足够证据回答 | 无支持页；全文搜索。未检出红烧肉或烹饪步骤；没有支持页，不能伪造引用。 |

## 复现与剩余限制

```bash
.venv/bin/python validation/verify_negative_case_evidence.py
```

该命令只读取本地 PDF，检查源文件指纹、全部摘录的原页归属、每个搜索式的命中页和历史 CSV 指纹；不调用模型。测试检查标签集完整性及历史数据不变，不能自动证明语义分类正确。

英文 13/15、中文 0/10 等检索数字仍只是历史 CSV 的结果，本轮未重新运行模型。没有调节阈值或替换模型。中文检索不足、生成层语义蕴含检查缺失、真实豆包生成未验证的问题仍然存在。完整的旧 Top-5 文本/逐题生成输出未保存，不能据当前整篇文档标签反推过去的生成效果。

最新检查与本地提交过程见 [CHECKPOINT_REPORT.md](CHECKPOINT_REPORT.md)。
