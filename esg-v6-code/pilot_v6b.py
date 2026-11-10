"""v6b: captions ARE context (user correction 2026-11-08).
Local context = narrative pool + caption, but source is tagged:
- narrative hit > 0 -> consistent
- caption-only hit -> 仅图注支撑 (weak support, counted as mismatch phenomenon, enters clustering)
- no hit -> mismatch, 无支撑
"""
import json, re, csv
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE="000516_output"
ctx=json.load(open(f"{BASE}/context_dynamic_v35.json"))
groups={g['group_id']:g for g in ctx['image_groups']}
res=[json.loads(l) for l in open(f"{BASE}/image_textualization_v1.1_output/image_textualization_results.jsonl", encoding="utf-8")]
struct={}
for l in res: struct[l['group_id']]=l.get('image_structured_data',[])
clip={r["image"][:12]: r for r in json.load(open("pilot_clip_v5.json")) if r.get("own_top_cos") is not None}
GENERIC={"国际医学","西安国际医学","医院","公司","集团","员工","护士","医生","患者","中心","建筑","外景","照片","图片","室外","室内"}
def core_words(strings):
    out=[]
    for s in strings or []:
        s=(s or "").strip()
        if not s: continue
        if len(s)>=4 and not any(g in s for g in ("国际医学","医院")):
            out.append(s)
        for m in re.finditer(r"[\u4e00-\u9fff]{2,6}", s):
            w=m.group(0)
            if w in GENERIC: continue
            if any(g in w for g in ("国际医学","医院")): continue
            out.append(w)
        for i in range(len(s)-1):
            w=s[i:i+2]
            if re.fullmatch(r"[\u4e00-\u9fff]{2}", w) and w not in GENERIC and "医院" not in w:
                out.append(w)
    return list(dict.fromkeys(out))
def norm(s):
    CN={'零':'0','一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','〇':'0'}
    return "".join(CN.get(ch,ch) for ch in (s or ""))
def time_hits(times, local):
    tot=0; hit=0; yo=0
    for t in times or []:
        t=norm(t)
        m=re.search(r"(\d{4})年?(\d{1,2})月?", t)
        if not m: continue
        tot+=1
        if m.group(2):
            if f"{m.group(1)}年{m.group(2)}月" in local or f"{m.group(1)}-{int(m.group(2)):02d}" in local or f"{m.group(1)}年{int(m.group(2))}月" in local:
                hit+=1
        else: yo+=1
    return hit, tot, yo

rows=[]
for gid in ["G0001","G0002","G0004","G0021","G0040","G0066"]:
    g=groups[gid]
    local=" ".join(c.get('text','') for c in g.get('context_pool',[]))
    caps=[]
    for im in g.get('images',[]):
        caps+= im.get('caption_candidates',[]) or []
    cap_text=" ".join(caps)
    local_blocks=len(g.get('context_pool',[]))
    for sd in struct.get(gid,[]):
        iid=sd.get('image_id','')
        itype=sd.get('Image Type') or ""
        is_cert=1 if ("证书" in itype or "牌匾" in itype or "铭牌" in itype) else 0
        ents=core_words(sd.get('Entity') or [])
        projs=core_words(sd.get('Project') or [])
        # narrative hits vs caption hits (per-image caption: this image's own caption candidates)
        my_caps=[]
        for im in g.get('images',[]):
            if im.get('filename','').startswith(iid[:12]):
                my_caps=im.get('caption_candidates',[]) or []
        my_cap_text=" ".join(my_caps)
        e_n=[e for e in ents if e in local]; p_n=[p for p in projs if p in local]
        e_c=[e for e in ents if e in my_cap_text]; p_c=[p for p in projs if p in my_cap_text]
        d_hit, d_tot, d_yo=time_hits(sd.get('Time') or [], local)
        c=clip.get(iid[:12], {})
        clip_score=c.get('own_top_cos')
        n_hit=len(e_n)+len(p_n); c_hit=len(e_c)+len(p_c)
        if n_hit>0:
            mismatch=0; rule="叙述支撑：一致"; support="叙述"
        elif c_hit>0:
            mismatch=1; rule="仅图注支撑：弱支撑，记为不匹配现象"; support="仅图注"
        elif local_blocks==0 and not my_caps:
            mismatch=1; rule="本地为空且无图注：无支撑"; support="无"
        else:
            mismatch=1; rule="叙述、图注均无命中：无支撑"; support="无"
        rows.append({"group":gid,"image_id":iid,"type":itype,"is_cert":is_cert,
            "local_blocks":local_blocks,"n_captions":len(my_caps),"caption":my_caps[0] if my_caps else "",
            "entity_n":f"{len(e_n)}/{len(ents)}","project_n":f"{len(p_n)}/{len(projs)}",
            "entity_c":f"{len(e_c)}/{len(ents)}","project_c":f"{len(p_c)}/{len(projs)}",
            "time":f"{d_hit}/{d_tot} (year_only={d_yo})","clip_own_top":clip_score,
            "mismatch":mismatch,"rule":rule,"support":support,"n_hit":n_hit,"c_hit":c_hit})
for r in rows:
    print(f"{r['group']} {r['image_id'][:12]} {r['type']} | 叙述 实体{r['entity_n']} 项目{r['project_n']} | 图注 实体{r['entity_c']} 项目{r['project_c']} | 本地{r['local_blocks']}段 图注{r['n_captions']}条 | {r['support']} | mismatch={r['mismatch']} ({r['rule']}) | 图注：{r['caption'][:30]}")
print(f"\n总计 {len(rows)} 张，一致 {sum(1 for r in rows if not r['mismatch'])} 张，不匹配 {sum(1 for r in rows if r['mismatch'])} 张，其中仅图注支撑 {sum(1 for r in rows if r['support']=='仅图注')} 张")
mis=[r for r in rows if r['mismatch']]
def feat(r):
    return [r["is_cert"], r["local_blocks"], 1 if r["support"]=="仅图注" else 0,
            r["clip_own_top"] if r["clip_own_top"] is not None else 0.0]
X=np.array([feat(r) for r in mis], float)
Xs=StandardScaler().fit_transform(X)
for k in [2,3,4]:
    km=KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    print(f"k={k} silhouette={silhouette_score(Xs, km.labels_):.3f}")
best_k, best_s = 3, -1
for k in [2,3,4]:
    km=KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    s=silhouette_score(Xs, km.labels_)
    if s>best_s: best_s=s; best_k=k
print(f"选 k={best_k}（轮廓系数最高 {best_s:.3f}）")
km=KMeans(n_clusters=best_k, random_state=0, n_init=10).fit(Xs)
hc=AgglomerativeClustering(n_clusters=best_k).fit(Xs)
clusters={}
for i,r in enumerate(mis):
    clusters.setdefault(int(km.labels_[i]), []).append(r)
    r["cluster"]=int(km.labels_[i])
for c in sorted(clusters):
    mem=clusters[c]
    print(f"\n簇 {c}：n={len(mem)}，支撑构成：{[m['support'] for m in mem]}")
    for m in mem:
        print(f"  {m['group']} {m['image_id'][:12]} {m['type']} | {m['rule']} | 图注：{m['caption'][:30]}")
json.dump(rows, open("pilot_v6b.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
with open("pilot_v6b.csv","w",newline="",encoding="utf-8-sig") as f:
    keys=["group","image_id","type","is_cert","local_blocks","n_captions","caption","entity_n","project_n","entity_c","project_c","time","clip_own_top","mismatch","rule","support","cluster"]
    w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows: w.writerow({k: r.get(k,"") for k in keys})
print("\nsaved pilot_v6b.json / pilot_v6b.csv")
