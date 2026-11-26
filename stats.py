import json,glob,numpy as np
def load(p): return json.load(open(glob.glob(p)[0]))
R={m:[load(f"*-{m}_s{s}.json") for s in range(3)] for m in ["resnet20","vit"]}
S={}
def ms(a): a=np.array(a,float); return [round(a.mean(),2), round(a.std(),2), [round(x,2) for x in a]]
for m in R:
    fp=[r["E1/fp"]["acc"] for r in R[m]]; S[f"{m}/fp"]=ms(fp)
    def calmean(r,k): return np.mean([r[f"{k}/s{c}"]["acc"] for c in range(3) if f"{k}/s{c}" in r])
    for k in ["E2/max/W8","E2/max/W6","E2/max/W4","E2/max/W2","E2/mse/W4","E2/mse/W2","E3/actonly/W32A8/s0","E3/actonly/W32A6/s0","E3/actonly/W32A4/s0","E3/plan/W8A8/s0","E3/plan/W6A6/s0","E3/plan/W4A4/s0"]:
        S[f"{m}/drop/{k}"]=ms([r["E1/fp"]["acc"]-r[k]["acc"] for r in R[m]])
        S[f"{m}/acc/{k}"]=ms([r[k]["acc"] for r in R[m]])
    for b in [8,6,4]:
        k=f"E3/improved/W{b}A{b}"
        S[f"{m}/drop/{k}"]=ms([r["E1/fp"]["acc"]-calmean(r,k) for r in R[m]])
        S[f"{m}/acc/{k}"]=ms([calmean(r,k) for r in R[m]])
        S[f"{m}/calsd/{k}"]=round(np.mean([np.std([r[f'{k}/s{c}']['acc'] for c in range(3)]) for r in R[m]]),2)
    for ab in [4,6]:
        for g in sorted({k.split('/')[3] for k in R[m][0] if k.startswith(f"E4/A{ab}/only/")}):
            for mode in ["only","allbut"]:
                S[f"{m}/E4/A{ab}/{mode}/{g}"]=ms([r["E1/fp"]["acc"]-r[f"E4/A{ab}/{mode}/{g}"]["acc"] for r in R[m]])
    for st in ["W8A4","W4A4","W6A6"]:
        base=None
        for f in ["none","log2","ptf","log2+ptf","ptf+linPTF","ptf+linPTF+geluPTF","mseW","log2+ptf+mseW"]:
            per=[np.mean([r[k]["acc"] for k in r if k.startswith(f"E6/{st}/{f}/s")]) for r in R[m] if any(k.startswith(f"E6/{st}/{f}/s") for k in r)]
            if not per: continue
            per=np.array(per)
            if f=="none": base=per
            S[f"{m}/E6/{st}/{f}"]={"acc":ms(per),"recov":ms(per-base),"drop":ms(np.array([r["E1/fp"]["acc"] for r in R[m]])-per)}
    # E4L mean drop per location
    names=[k[7:] for k in R[m][0] if k.startswith("E4L/A4/")]
    S[f"{m}/E4L"]={n:ms([r["E4L/base"]["acc"]-r["E4L/A4/"+n]["acc"] for r in R[m]]) for n in names}
json.dump(S,open("final_stats.json","w"),indent=1)
for k,v in S.items():
    if "E4L" in k: continue
    print(k,v)
