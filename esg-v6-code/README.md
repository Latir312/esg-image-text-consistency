# ESG 图文一致性评估方法 v6 — 代码包（完整版，含数据与模型）

- 日期：2026-11-08（v6c 为最终口径）
- 主题：ESG 年报图片与本地上下文不匹配；最终目标是按不匹配的类型聚类，先记录现象、后聚类、再按大类命名。

## 口径（最终版 v6c）

- 只比本地上下文（叙述段 + 图注）；图注就是上下文，图注支撑算一致。
- 比较不用大模型当裁判；照片主题级，证书、图表字段级；生产零人工（十万多份年报）。
- 匹配程度是一组分字段率，不合成一个总分；是否不匹配靠规则判定。

## 包里有什么

- `esg-v6-code/`：全部代码、方法稿、试点报告、v6c 结果数据、总流程图、README（见该目录下的 README.md）。
- `000516_output/`：试点数据，国际医学 000516 2024 年 ESG 报告 MinerU 解析结果（原 PDF、full.md、content_list.json、context_dynamic_v35.json、images_clean 等）。
- `clip-onnx/`：中文 CLIP base 量化模型（Xenova/chinese-clip-vit-base-patch16，`model_quantized.onnx` 183MB）与 `tokenizer.json`、`config.json`、`preprocessor_config.json`；调用脚本在 `esg-v6-code/clip-onnx/run_clip_v5.py`。

## 运行

```bash
pip install onnxruntime scikit-learn tokenizers pillow python-docx matplotlib
cd esg-v6-code
python3 pilot_v6c.py
python3 clip_zero_shot_classify.py
python3 build_report_docx.py
python3 build_flowchart.py
```

注意：`pilot_v6c.py` 与 CLIP 脚本默认在上级目录找 `000516_output/` 与 `clip-onnx/`，即按本包解压后的目录结构直接运行即可。
