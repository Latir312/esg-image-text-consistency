"""Clustering demo: record mismatch PHENOMENON as features, NO type label upfront.
Then cluster, then name clusters post-hoc. Pilot only: 23 images, 20 mismatch instances.
Features are all observable, no human type judgment.
"""
import json, numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

v5 = json.load(open("pilot_v5.json"))
clip = {r["group"]+"|"+r["label"]: r for r in json.load(open("pilot_clip_v5.json"))}
# local blocks per group
ctx = json.load(open("000516_output/context_dynamic_v35.json"))
local_n = {g['group_id']: len(g.get('context_pool', [])) for g in ctx['image_groups']}
# build features per image in v5 (23 rows)
rows = []
for r in v5:
    gid = r["group"]
    is_cert = 1 if r["route"].startswith("B") else 0
    is_photo = 1 if r["route"].startswith("C") or "composite" in r["route"] else 0
    # entity hits from detail string
    import re
    m = re.search(r"entity_hits=\[(.*?)\]", r["detail"])
    ent_hits = 0
    if m and m.group(1).strip():
        ent_hits = len([x for x in m.group(1).split(",") if x.strip()])
    # clip own top for photos: look up by group+label approx
    clip_score = 0.0
    for k, c in clip.items():
        if k.startswith(gid+"|") and c.get("own_top_cos") is not None:
            # match by image? clip json has image filename, v5 image_id is full id
            if c["image"][:12] == r["image_id"][:12]:
                clip_score = c["own_top_cos"]
    # image has date? from pilot_v5 detail / award: use v5 route B date_hit in detail
    m2 = re.search(r"date_hit=\[(.*?)\]", r["detail"])
    date_hit = 0
    if m2 and m2.group(1).strip():
        date_hit = 1
    # consistent? v5 class contains 主题一致
    is_consistent = 1 if "主题一致" in r["class"] else 0
    rows.append({
        "group": gid, "image": r["image_id"][:12], "route": r["route"], "class_v5": r["class"],
        "is_cert": is_cert, "is_photo": is_photo, "local_blocks": local_n.get(gid, 0),
        "entity_hits": ent_hits, "clip_own": clip_score, "date_hit": date_hit,
        "is_consistent": is_consistent,
    })
# mismatch instances only (exclude 3 consistent)
mis = [r for r in rows if not r["is_consistent"]]
print(f"total {len(rows)} images, mismatch instances {len(mis)} (consistent excluded: {[r['group']+'/'+r['image'] for r in rows if r['is_consistent']]})")
print("\n--- recorded phenomenon features (no type label) ---")
for r in mis:
    print(f"{r['group']} {r['image']} cert={r['is_cert']} photo={r['is_photo']} local_blocks={r['local_blocks']} entity_hits={r['entity_hits']} clip={r['clip_own']:.3f} date_hit={r['date_hit']}")
X = np.array([[r["is_cert"], r["is_photo"], r["local_blocks"], r["entity_hits"], r["clip_own"], r["date_hit"]] for r in mis], dtype=float)
Xs = StandardScaler().fit_transform(X)
for k in [3]:
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    print(f"\n--- KMeans k={k} (features standardized, random_state=0) ---")
    for c in range(k):
        members = [mis[i] for i in range(len(mis)) if km.labels_[i]==c]
        print(f"\nCluster {c}: n={len(members)}")
        for m in members:
            print(f"  {m['group']} {m['image']} cert={m['is_cert']} local_blocks={m['local_blocks']} entity_hits={m['entity_hits']} clip={m['clip_own']:.3f} | v5 class was: {m['class_v5']}")
        # cluster centre in original units (approx, via mean of members)
        import numpy as _np
        arr = _np.array([[x['is_cert'], x['local_blocks'], x['entity_hits'], x['clip_own']] for x in members])
        print(f"  centre (mean cert, local_blocks, entity_hits, clip): {arr.mean(axis=0).round(2)}")
json.dump({"mismatch_n": len(mis), "rows": rows}, open("pilot_cluster_demo.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nsaved pilot_cluster_demo.json")
