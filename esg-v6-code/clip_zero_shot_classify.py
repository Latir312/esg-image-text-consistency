"""CLIP zero-shot classification test on the 23 pilot images.
Labels: 证书 / 照片 / 图表 / 复合图. Ground truth by manual type:
G0004 (16) = 证书, G0001/G0002/G0040/G0066 (6) = 照片, G0021 (1) = 复合图.
"""
import json, numpy as np
from PIL import Image
import onnxruntime as ort
from tokenizers import Tokenizer

BASE="000516_output"
FILES=[
 ("G0001","d894414252869d9598fd58ff3912dc851c8727d3be19092b817e8707892cd200.jpg","照片"),
 ("G0002","0796e72a296a369d6cf04b5f453836b664ffdf4f4acca7da342a9018d8113d04.jpg","照片"),
 ("G0002","aaf85ad678624a48a046dfaf311f1b897af391c95181dd3db94578349d7a6bd1.jpg","照片"),
 ("G0021","c42397d314a467c7b38639b79cb8bec2dee4246bdbea5ec9915c3abf0f860c0f.jpg","复合图"),
 ("G0040","16027a267eb1969b240441277fd504803a1303a2c167fe02267ed63b04fe7dd6.jpg","照片"),
 ("G0066","f3396384429bb1aa8433be7ed88ca4f7a94171fc59ad5c958fb2b1896a6161e5.jpg","照片"),
 ("G0066","7ac4cd947fa0b4c4855346ca311d57df7e04ec56a4a17edfc730c9c6bedf4e12.jpg","照片"),
]
# add G0004 certs
ctx=json.load(open(f"{BASE}/context_dynamic_v35.json"))
for g in ctx['image_groups']:
    if g['group_id']=="G0004":
        for im in g['images']:
            FILES.append(("G0004", im['filename'], "证书"))
LABELS=["一张证书","一张照片","一张图表","一张照片和图表拼接成的复合图"]
tok=Tokenizer.from_file("clip-onnx/tokenizer.json")
tok.enable_padding(length=77); tok.enable_truncation(max_length=77)
so=ort.SessionOptions(); so.intra_op_num_threads=2; so.inter_op_num_threads=1
sess=ort.InferenceSession("clip-onnx/model_quantized.onnx", so, providers=["CPUExecutionProvider"])
MEAN=np.array([0.48145466,0.4578275,0.40821073],dtype=np.float32)
STD=np.array([0.26862954,0.26130258,0.27577711],dtype=np.float32)
def enc_texts(texts):
    encs=[tok.encode(t) for t in texts]
    return np.array([e.ids for e in encs],dtype=np.int64), np.array([e.attention_mask for e in encs],dtype=np.int64)
def enc_img(path):
    img=Image.open(path).convert("RGB").resize((224,224),Image.BICUBIC)
    arr=np.asarray(img,dtype=np.float32)/255.0
    return ((arr-MEAN)/STD).transpose(2,0,1)[None,:,:,:].astype(np.float32)
ids,mask=enc_texts(LABELS)
out=sess.run(["text_embeds"],{"input_ids":ids,"attention_mask":mask,"pixel_values":np.zeros((1,3,224,224),dtype=np.float32)})[0]
t_emb=out/np.linalg.norm(out,axis=-1,keepdims=True)
correct=0
rows=[]
for gid,fname,truth in FILES:
    o=sess.run(["image_embeds"],{"input_ids":ids[:1],"attention_mask":mask[:1],"pixel_values":enc_img(f"{BASE}/images_clean/{fname}")})[0]
    i_emb=o/np.linalg.norm(o,axis=-1,keepdims=True)
    sims=(i_emb@t_emb.T)[0]
    pred=["证书","照片","图表","复合图"][int(np.argmax(sims))]
    ok = pred==truth
    correct+=ok
    rows.append({"group":gid,"image":fname[:12],"truth":truth,"pred":pred,"sims":[round(float(s),3) for s in sims],"ok":ok})
    print(f"{gid} {fname[:12]} 真实={truth} 预测={pred} {'✓' if ok else '✗'} 相似度 证书/照片/图表/复合={ [round(float(s),3) for s in sims] }")
print(f"\n准确率 {correct}/{len(FILES)} = {correct/len(FILES):.2f}")
json.dump(rows, open("clip_zero_shot_classify.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
