"""Chinese CLIP (Xenova/chinese-clip-vit-base-patch16, quantized ONNX) — Route C test.
Runs locally on this VM with onnxruntime CPU, no torch, no GPU.
Local pool only (context_dynamic_v35). Candidates = sentences from local pool.
Controls = local sentences from the OTHER 5 groups, to test discriminativeness.
"""
import json, re
import numpy as np
from PIL import Image
import onnxruntime as ort
from tokenizers import Tokenizer

BASE = "000516_output"
ctx = json.load(open(f"{BASE}/context_dynamic_v35.json"))
groups = {g['group_id']: g for g in ctx['image_groups']}

def split_sentences(text):
    parts = re.split(r"(?<=[。！？；])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]

local_sents = {}
for gid, g in groups.items():
    sents = []
    for c in g.get('context_pool', []):
        sents.extend(split_sentences(c.get('text', '')))
    seen = set(); out = []
    for s in sents:
        if s not in seen:
            seen.add(s); out.append(s)
    local_sents[gid] = out

PHOTOS = [
    ("G0001", "d894414252869d9598fd58ff3912dc851c8727d3be19092b817e8707892cd200.jpg", "航拍照片"),
    ("G0002", "0796e72a296a369d6cf04b5f453836b664ffdf4f4acca7da342a9018d8113d04.jpg", "建筑外景-妇女儿童/骨科"),
    ("G0002", "aaf85ad678624a48a046dfaf311f1b897af391c95181dd3db94578349d7a6bd1.jpg", "建筑外景-高新医院"),
    ("G0021", "c42397d314a467c7b38639b79cb8bec2dee4246bdbea5ec9915c3abf0f860c0f.jpg", "建筑照片（复合图，含图例）"),
    ("G0040", "16027a267eb1969b240441277fd504803a1303a2c167fe02267ed63b04fe7dd6.jpg", "建筑外景-血液病/肿瘤"),
    ("G0066", "f3396384429bb1aa8433be7ed88ca4f7a94171fc59ad5c958fb2b1896a6161e5.jpg", "培训听讲"),
    ("G0066", "7ac4cd947fa0b4c4855346ca311d57df7e04ec56a4a17edfc730c9c6bedf4e12.jpg", "集体宣誓"),
]

tok = Tokenizer.from_file("clip-onnx/tokenizer.json")
tok.enable_padding(length=77)
tok.enable_truncation(max_length=77)
so = ort.SessionOptions()
so.intra_op_num_threads = 2
so.inter_op_num_threads = 1
sess = ort.InferenceSession("clip-onnx/model_quantized.onnx", so, providers=["CPUExecutionProvider"])
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def encode_texts(texts):
    encs = [tok.encode(t) for t in texts]
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    return ids, mask

def encode_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

def cos(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T

results = []
for gid, fname, label in PHOTOS:
    path = f"{BASE}/images_clean/{fname}"
    own = local_sents.get(gid, [])
    controls = []
    for og, ss in local_sents.items():
        if og != gid:
            controls.extend([(og, s) for s in ss])
    if not own:
        print(f"\n===== {gid} {label} =====\n  NO LOCAL SENTENCES — cannot compute CLIP vs local")
        results.append({"group": gid, "image": fname, "label": label, "own_top": None, "control_top": None, "note": "no local context"})
        continue
    texts = own + [s for _, s in controls]
    ids, mask = encode_texts(texts)
    pix = encode_image(path)
    out = sess.run(["image_embeds", "text_embeds"], {"pixel_values": pix, "input_ids": ids, "attention_mask": mask})
    img_emb, txt_emb = out[0], out[1]
    sims = cos(img_emb, txt_emb)[0]
    own_sims = sims[:len(own)]
    ctrl_sims = sims[len(own):]
    order_own = np.argsort(-own_sims)
    order_ctrl = np.argsort(-ctrl_sims)
    print(f"\n===== {gid} {label} =====")
    print(f"  own local sentences: {len(own)}, controls: {len(controls)}")
    for rank in order_own[:3]:
        print(f"  OWN {rank+1}. cos={own_sims[rank]:.3f} | {own[rank][:120]}")
    for rank in order_ctrl[:3]:
        og, s = controls[rank]
        print(f"  CTRL {rank+1}. cos={ctrl_sims[rank]:.3f} [{og}] {s[:120]}")
    print(f"  own top={own_sims[order_own[0]]:.3f}  control top={ctrl_sims[order_ctrl[0]]:.3f}  margin(own-control)={own_sims[order_own[0]]-ctrl_sims[order_ctrl[0]]:+.3f}")
    results.append({
        "group": gid, "image": fname, "label": label,
        "own_top": [{"cos": round(float(own_sims[i]), 3), "text": own[i]} for i in order_own[:3]],
        "control_top": [{"cos": round(float(ctrl_sims[i]), 3), "group": controls[i][0], "text": controls[i][1]} for i in order_ctrl[:3]],
        "own_top_cos": round(float(own_sims[order_own[0]]), 3),
        "control_top_cos": round(float(ctrl_sims[order_ctrl[0]]), 3),
        "margin": round(float(own_sims[order_own[0]] - ctrl_sims[order_ctrl[0]]), 3),
    })
json.dump(results, open("pilot_clip_v5.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nsaved pilot_clip_v5.json")
