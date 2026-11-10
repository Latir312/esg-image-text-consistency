import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.font_manager import FontProperties

fp = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", size=13)
fp_b = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", size=15)
fp_s = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", size=11)

fig, ax = plt.subplots(figsize=(11, 15))
ax.set_xlim(0, 11); ax.set_ylim(0, 15); ax.axis("off")

def box(x, y, w, h, title, sub="", color="#E8F0FE", edge="#2F5597"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.18",
                                facecolor=color, edgecolor=edge, linewidth=1.6))
    ax.text(x+w/2, y+h/2 + (0.22 if sub else 0), title, ha="center", va="center",
            fontproperties=fp_b, color="#1F3864")
    if sub:
        _fp = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                             size=11 if len(sub) <= 22 else 9)
        ax.text(x+w/2, y+h/2 - 0.42, sub, ha="center", va="center",
                fontproperties=_fp, color="#333333")

def arrow(x1, y1, x2, y2, text=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color="#2F5597"))
    if text:
        ax.text((x1+x2)/2 + 0.15, (y1+y2)/2, text, fontproperties=fp_s, color="#2F5597")

def diamond(cx, cy, w, h, title):
    pts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="#FFF2CC", edgecolor="#BF8F00", linewidth=1.6))
    ax.text(cx, cy+0.12, title, ha="center", va="center", fontproperties=fp_b, color="#7F6000")
    ax.text(cx, cy-0.32, "规则判定", ha="center", va="center", fontproperties=fp_s, color="#7F6000")

ax.text(5.5, 14.6, "ESG 年报图文不匹配评估 · 总流程图（方法 v6c）", ha="center", fontproperties=fp_b, fontsize=19, color="#1F3864")
ax.text(5.5, 14.15, "核心就三段：分流 → 比对 → 聚类，其余都是支撑", ha="center", fontproperties=fp, color="#555555")

# main chain (center x=4.2, width 5.2)
box(1.6, 12.7, 5.2, 1.0, "① 输入：ESG 年报 PDF", "最终十万多份，批量跑")
arrow(4.2, 12.7, 4.2, 12.45)
box(1.6, 11.45, 5.2, 1.0, "② 解析：MinerU 解析", "图片 + 本地池（叙述 + 图注）冻结")
arrow(4.2, 11.45, 4.2, 11.2)
box(1.6, 10.2, 5.2, 1.0, "③ 分流：照片 / 证书 / 图表 / 复合图", "MinerU类型 → 图注关键词 → VLM → CLIP → 复合图检查")
arrow(4.2, 10.2, 4.2, 9.95)
box(1.6, 8.95, 5.2, 1.0, "④ 结构化提取", "图用 VLM 提取，本地用规则提取")
arrow(4.2, 8.95, 4.2, 8.7)
box(1.6, 7.7, 5.2, 1.0, "⑤ 分字段匹配程度", "主体率 / 项目率 / 时间率 / 数值率 / CLIP分 / 限定保留率")
ax.text(4.2, 7.45, "不合成一个总分", ha="center", fontproperties=fp_s, color="#C00000")
arrow(4.2, 7.45, 4.2, 7.2)
diamond(4.2, 6.45, 4.6, 1.3, "⑥ 是否不匹配？")
# branches
ax.text(1.15, 6.45, "否", ha="center", fontproperties=fp_b, color="#2E7D32")
ax.text(7.25, 6.45, "是", ha="center", fontproperties=fp_b, color="#C00000")
# left: consistent
ax.add_patch(FancyArrowPatch((1.9, 6.45), (0.9, 6.45), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#2E7D32"))
ax.add_patch(FancyArrowPatch((0.9, 6.45), (0.9, 5.3), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#2E7D32"))
box(0.15, 4.3, 1.9, 1.0, "一致", "", color="#E2EFDA", edge="#2E7D32")
# right: mismatch chain
ax.add_patch(FancyArrowPatch((6.5, 6.45), (8.6, 6.45), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#C00000"))
ax.add_patch(FancyArrowPatch((8.6, 6.45), (8.6, 5.3), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#C00000"))
box(7.45, 4.3, 2.3, 1.0, "⑦ 不匹配", "记录现象特征向量", color="#FCE4E4", edge="#C00000")
arrow(8.6, 4.3, 8.6, 4.05)
box(7.45, 3.05, 2.3, 1.0, "⑧ 无监督聚类", "KMeans + 层次对照，k 轮廓系数选")
arrow(8.6, 3.05, 8.6, 2.8)
box(7.45, 1.8, 2.3, 1.0, "⑨ 聚完命名大类", "看簇中心再起名字")
# converge to summary
ax.add_patch(FancyArrowPatch((1.1, 4.3), (1.1, 1.35), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#2E7D32"))
ax.add_patch(FancyArrowPatch((8.6, 1.8), (8.6, 1.35), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color="#C00000"))
ax.add_patch(FancyArrowPatch((8.6, 1.35), (6.0, 1.35), arrowstyle="-", mutation_scale=16, linewidth=1.6, color="#2F5597"))
ax.add_patch(FancyArrowPatch((1.1, 1.35), (2.4, 1.35), arrowstyle="-", mutation_scale=16, linewidth=1.6, color="#2F5597"))
box(2.4, 0.35, 3.6, 1.0, "⑩ 汇总：公司–年面板", "不匹配率、各簇占比")
# side box: calibration
ax.add_patch(FancyBboxPatch((8.15, 9.3), 2.7, 2.6, boxstyle="round,pad=0.12,rounding_size=0.18",
                            facecolor="#F2F2F2", edgecolor="#808080", linewidth=1.4, linestyle=(0, (4, 3))))
ax.text(9.5, 11.55, "开发阶段一次", ha="center", fontproperties=fp_b, color="#333333")
ax.text(9.5, 10.95, "小样本人工标定", ha="center", fontproperties=fp_s)
ax.text(9.5, 10.55, "（500–1000 张）", ha="center", fontproperties=fp_s)
ax.text(9.5, 10.15, "冻住阈值 / 词表", ha="center", fontproperties=fp_s)
ax.text(9.5, 9.75, "/ 分类规则", ha="center", fontproperties=fp_s)
ax.text(9.5, 9.45, "生产零人工", ha="center", fontproperties=fp_b, color="#C00000")
ax.add_patch(FancyArrowPatch((8.15, 10.7), (6.85, 10.7), arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.4, color="#808080", linestyle=(0, (4, 3))))
plt.savefig("flowchart_v6c.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved flowchart_v6c.png")
