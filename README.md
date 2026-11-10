# ESG 年报图片与本地上下文一致性评估方法 v6

研究问题：ESG 年报里的图片和它旁边的文字对不上，是年报本身的手法，还是我们的方法问题？本仓库是方法 v6 的代码、方法稿与试点结果。

## 方法（一句话）

结构化字段逐项比对，找出不匹配、记录现象特征；不预先判类型，用无监督聚类把不匹配聚成大类，聚完再命名。

## 口径（v6c，最终版）

- 只比本地上下文，不比全文；本地上下文 = 叙述段 + 图注，图注就是上下文，图注支撑算一致。
- 比较不用大模型当裁判；模型只做提取，比对用规则。
- 照片只比到主题级，证书、图表比到字段级；复合图先拆，拆不开就排除并单独报比例。
- 匹配程度是一组分字段率，不合成一个总分；是否不匹配靠规则判定。
- 生产审核十万多份年报须全自动，零人工；人工只在开发阶段做一次小样本标定。

## 仓库内容

- `esg-v6-code/pilot_v6c.py`：最终版比对脚本（v6c 口径）。
- `esg-v6-code/pilot_v6.py`：旧口径版，仅供对照。
- `esg-v6-code/pilot_v6b.py`：已撤回的版本（曾误把“仅图注支撑”当弱支撑），不要用。
- `esg-v6-code/pilot_cluster_demo.py`：聚类演示脚本。
- `esg-v6-code/clip_zero_shot_classify.py`：CLIP 零样本分类测试。
- `esg-v6-code/clip-onnx/run_clip_v5.py`：CLIP 本地实测脚本。
- `esg-v6-code/build_report_docx.py`、`build_flowchart.py`：汇报文档与流程图生成脚本。
- `esg-v6-code/method_v6_recommended.md`：方法稿 v6（含第 6 节匹配程度、第 8 节图注口径最终版、第 9 节匹配前分类）。
- `esg-v6-code/pilot_report_v6.md`：试点报告。
- `esg-v6-code/pilot_v6c.json`、`.csv`：试点 v6c 结果数据。

## 试点结果（国际医学 000516，2024 年报，23 张图）

一致 19 张（叙述支撑 4 张 + 图注支撑 15 张），不匹配 4 张。不匹配太少，试点演示不了聚类，等大样本。

## 本仓库没有的东西（请另外获取）

- 试点数据 `000516_output/`（原 PDF、MinerU 解析结果、本地池、图片）：数据量较大，未随仓库上传。
- 中文 CLIP 模型：本仓库只有调用脚本，权重请从 HuggingFace 公开模型 `Xenova/chinese-clip-vit-base-patch16` 下载（`model_quantized.onnx`、`tokenizer.json`、`config.json`、`preprocessor_config.json`），放到 `clip-onnx/` 目录下即可运行。注意：CLIP 实测只能当粗筛，实体重合仍是主信号。
- 汇报文档 docx 与流程图 png：由上面两个生成脚本重新生成。
