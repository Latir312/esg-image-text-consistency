"""v6 pilot: field-by-field matching-degree profile (rules only), mismatch by rule,
record phenomenon features (no type label), then unsupervised clustering, name after.
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
    """split structured strings into 2-6 char Chinese words, drop generic, dedupe.
    Also keep full distinctive phrases >=4 chars that are not generic."""
    out=[]
    for s in strings or []:
        s=(s or "").strip()
        if not s: continue
        # full phrase if distinctive
        if len(s)>=4 and not any(g in s for g in ("国际医学","医院")):
            out.append(s)
        for m in re.finditer(r"[\u4e00-\u9fff]{2,6}", s):
            w=m.group(0)
            if w in GENERIC: continue
            if any(g in w for g in ("国际医学","医院")): continue
            out.append(w)
        # 2-char sliding windows: 培训/集体/高新/宣誓 must be findable even inside a long phrase
        for i in range(len(s)-1):
            w=s[i:i+2]
            if re.fullmatch(r"[\u4e00-\u9fff]{2}", w) and w not in GENERIC and "医院" not in w:
                out.append(w)
    # drop words that are substrings of a longer kept phrase? keep all, dedupe
    return list(dict.fromkeys(out))

def norm(s):
    CN={'零':'0','一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','〇':'0'}
    return "".join(CN.get(ch,ch) for ch in (s or ""))
def time_hits(times, local):
    """return (hits, total, year_only_count). Year-only counts in total, hit only if year+month present."""
    tot=0; hit=0; yo=0
    for t in times or []:
        t=norm(t)
        m=re.search(r"(\d{4})年?(\d{1,2})月?", t)
        if not m: continue
        tot+=1
        if m.group(2):
            pat=f"{m.group(1)}年{m.group(2)}月"
            pat2=f"{m.group(1)}-{int(m.group(2)):02d}"
            if pat in local or pat2 in local or f"{m.group(1)}年{int(m.group(2))}月" in local:
                hit+=1
        else:
            yo+=1
    return hit, tot, yo

QUAL=["全公司","某工厂","全年","试点","已实现","目标","计划","仅限","部分"]
rows=[]
for gid in ["G0001","G0002","G0004","G0021","G0040","G0066"]:
    local=" ".join(c.get('text','') for c in groups[gid].get('context_pool',[]))
    local_blocks=len(groups[gid].get('context_pool',[]))
    quals=[q for q in QUAL if q in local]
    for sd in struct.get(gid,[]):
        iid=sd.get('image_id','')
        itype=sd.get('Image Type') or ""
        is_cert=1 if ("证书" in itype or "牌匾" in itype or "铭牌" in itype) else 0
        is_photo=1 if not is_cert else 0
        ents=core_words(sd.get('Entity') or [])
        projs=core_words(sd.get('Project') or [])
        e_hit=[e for e in ents if e in local]
        p_hit=[p for p in projs if p in local]
        d_hit, d_tot, d_yo=time_hits(sd.get('Time') or [], local)
        c=clip.get(iid[:12], {})
        clip_score=c.get('own_top_cos')
        # mismatch by RULE (no total score):
        hard_total=len(ents)+len(projs)
        hard_hit=len(e_hit)+len(p_hit)
        if local_blocks==0:
            mismatch=1; rule="本地为空：无支撑"
        elif hard_total>0 and hard_hit==0:
            mismatch=1; rule="硬字段（主体/项目）一个都没出现：无支撑"
        elif is_photo and hard_hit==0 and (clip_score is not None and clip_score<0.40):
            mismatch=1; rule="照片主题词无命中且CLIP低：主题不匹配"
        elif hard_hit>0:
            mismatch=0; rule="硬字段有命中：一致（主题级/字段级）"
        else:
            mismatch=1; rule="无硬字段可比：不确定，记为不匹配（待复核）"
        # qualifier retention: NA in pilot (no qualifier in local for these groups, except check)
        rows.append({
            "group":gid,"image_id":iid,"type":itype,"is_cert":is_cert,"is_photo":is_photo,
            "local_blocks":local_blocks,"entity_rate":round(len(e_hit)/len(ents),2) if ents else None,
            "entity":f"{len(e_hit)}/{len(ents)}","project_rate":round(len(p_hit)/len(projs),2) if projs else None,
            "project":f"{len(p_hit)}/{len(projs)}","time_rate":round(d_hit/d_tot,2) if d_tot else None,
            "time":f"{d_hit}/{d_tot} (year_only={d_yo})","numeric_rate":None,"qualifier_rate":None,
            "clip_own_top":clip_score,"qualifiers_in_local":quals,
            "mismatch":mismatch,"rule":rule,
            "hard_hit":hard_hit,"hard_total":hard_total,
            "entity_hits":e_hit,"project_hits":p_hit,
        })
for r in rows:
    print(f"{r['group']} {r['image_id'][:12]} {r['type']} | 实体 {r['entity']} 项目 {r['project']} 时间 {r['time']} | CLIP={r['clip_own_top']} | 本地={r['local_blocks']}段 | mismatch={r['mismatch']} ({r['rule']}) | 命中词={r['entity_hits']+r['project_hits']}")
print(f"\n总计 {len(rows)} 张，一致 {sum(1 for r in rows if not r['mismatch'])} 张，不匹配 {sum(1 for r in rows if r['mismatch'])} 张")
# --- clustering: mismatch only, features = phenomenon only, no type label ---
mis=[r for r in rows if r['mismatch']]
def feat(r):
    return [r["is_cert"], r["is_photo"], r["local_blocks"],
            r["entity_rate"] if r["entity_rate"] is not None else 0.0,
            r["project_rate"] if r["project_rate"] is not None else 0.0,
            r["time_rate"] if r["time_rate"] is not None else 0.0,
            r["clip_own_top"] if r["clip_own_top"] is not None else 0.0]
X=np.array([feat(r) for r in mis], float)
Xs=StandardScaler().fit_transform(X)
print(f"\n聚类：不匹配 {len(mis)} 张，特征=[is_cert,is_photo,local_blocks,entity_rate,project_rate,time_rate,clip]")
for k in [2,3,4]:
    if k>=len(mis): continue
    km=KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    print(f"k={k} silhouette={silhouette_score(Xs, km.labels_):.3f}")
best_k=2
best_s=-1
for k in [2,3,4]:
    km=KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    s=silhouette_score(Xs, km.labels_)
    if s>best_s: best_s=s; best_k=k
print(f"选 k={best_k}（轮廓系数最高 {best_s:.3f}）")
km=KMeans(n_clusters=best_k, random_state=0, n_init=10).fit(Xs)
# hierarchical check
hc=AgglomerativeClustering(n_clusters=best_k).fit(Xs)
print(f"层次聚类 k={best_k} 与 KMeans 一致率：{(hc.labels_==km.labels_).mean() if best_k==2 else '标签顺序不同，直接看成员'}")
clusters={}
for i,r in enumerate(mis):
    clusters.setdefault(int(km.labels_[i]), []).append(r)
    r["cluster"]=int(km.labels_[i])
for c in sorted(clusters):
    mem=clusters[c]
    arr=np.array([[m["is_cert"],m["local_blocks"],m["entity_rate"] or 0,m["project_rate"] or 0,m["clip_own_top"] or 0] for m in mem])
    print(f"\n簇 {c}：n={len(mem)}，中心均值 [证书,本地段数,实体率,项目率,CLIP]={arr.mean(axis=0).round(2)}")
    for m in mem:
        print(f"  {m['group']} {m['image_id'][:12]} {m['type']} 实体{m['entity']} 项目{m['project']} 时间{m['time']} CLIP={m['clip_own_top']} | {m['rule']}")
    print(f"  层次聚类成员：{[ (m['group'], m['image_id'][:12]) for m in mem]}")
json.dump(rows, open("pilot_v6.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
with open("pilot_v6.csv","w",newline="",encoding="utf-8-sig") as f:
    keys=["group","image_id","type","is_cert","is_photo","local_blocks","entity","entity_rate","project","project_rate","time","time_rate","numeric_rate","qualifier_rate","clip_own_top","mismatch","rule","cluster"]
    w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows: w.writerow({k: r.get(k,"") for k in keys})
print("\nsaved pilot_v6.json / pilot_v6.csv")
