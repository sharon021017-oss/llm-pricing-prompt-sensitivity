import glob,json,re,numpy as np,pandas as pd,sys
from scipy import stats
ARMS=["P1","P1-trunc","P1-para"]; rng=np.random.default_rng(1)
rows=[]; plans=[]
for f in glob.glob("results/duopoly_*_run*.jsonl"):
    L=[json.loads(l) for l in open(f)]
    if len(L)<100: continue
    arm,run,al=L[0]["arm"],L[0]["run"],L[0]["alpha"]
    P=np.array([[a["price"]/r["alpha"] for a in r["agents"]] for r in L])
    Pi=np.array([[a["profit"]/r["alpha"] for a in r["agents"]] for r in L])
    w=slice(50,100); pm=P.mean(1)
    d=np.diff(pm)
    rows.append({"arm":arm,"run":run,
      "profit":Pi[w].mean(), "joint_profit":Pi[w].sum(1).mean(),
      "firm_gap":np.abs(P[w,0]-P[w,1]).mean(),
      "profit_gap":np.abs(Pi[w,0]-Pi[w,1]).mean(),
      "changes":(np.abs(np.diff(P[w],axis=0))>0.005).mean(),       # share of firm-periods with a price change
      "big_cuts":(np.diff(P[w],axis=0)< -0.05).sum(),               # firm-level cuts >0.05 in terminal window
      "war_episodes":int(((d<-0.03)[49:]).sum()) if False else int((np.diff(pm[w])< -0.05).sum()),
      "undercut_share":(np.abs(P[w,0]-P[w,1])>0.01).mean()})
    for r in L:
        for a in r["agents"]:
            plans.append({"arm":arm,"run":run,"t":r["period"],"text":(a["plans"]+" "+a["insights"]+" "+a.get("observations","")).lower()})
R=pd.DataFrame(rows); R["arm"]=pd.Categorical(R.arm,ARMS)
print(R.groupby("arm",observed=True)[["profit","joint_profit","firm_gap","profit_gap","changes","big_cuts","war_episodes","undercut_share"]].mean().round(4).T)
b=R[R.arm=="P1"]
for arm in ["P1-trunc","P1-para"]:
    g=R[R.arm==arm]
    for v in ["profit","joint_profit","firm_gap","profit_gap","changes","big_cuts"]:
        print(arm,v,"diff",round(g[v].mean()-b[v].mean(),4),"welch p",round(stats.ttest_ind(g[v],b[v],equal_var=False).pvalue,4),"MW p",round(stats.mannwhitneyu(g[v],b[v]).pvalue,4))
R.to_csv("results/v2_profit_stability.csv",index=False)
# plans text
T=pd.DataFrame(plans)
kw={"undercut":r"undercut","price war":r"price war|pricing war","aggressive":r"aggressiv","explore/test":r"\bexplor|\btest",
    "match competitor":r"\bmatch","collude/cooperate":r"collu|cooperat|tacit","long run/term":r"long[- ](run|term)",
    "maximi[sz]e profit":r"maximi[sz]\w* (the )?(user'?s )?profit","volume/market share":r"volume|market share",
    "stabil":r"stabil","higher price":r"higher price|raise|increase (our )?price","lower price":r"lower price|reduc\w* (our )?price|cut"}
out=[]
for k,pat in kw.items():
    T[k]=T.text.str.contains(pat,regex=True)
    runlvl=T.groupby(["arm","run"])[k].mean().reset_index()
    m=runlvl.groupby("arm")[k].mean()
    p1=runlvl[runlvl.arm=="P1"][k]
    row={"term":k,**{a:round(100*m[a],1) for a in ARMS}}
    for a in ["P1-trunc","P1-para"]:
        row[a+"_p"]=round(stats.mannwhitneyu(runlvl[runlvl.arm==a][k],p1).pvalue,4)
    out.append(row)
K=pd.DataFrame(out); print("\n% of agent-periods mentioning (run-level average):\n",K.to_string(index=False))
K.to_csv("results/v2_plans_keywords.csv",index=False)
T["len"]=T.text.str.len()
print("\ntext length", T.groupby("arm").len.mean().round(0).to_dict())
