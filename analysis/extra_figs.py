import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="results/"; OUT="results/"
ARMS=["P1","P1-trunc","P1-para"]
cols={"P1":"#3b6fb6","P1-trunc":"#c0504d","P1-para":"#4f9a5b"}
PN,PM,P1M=1.473,1.925,1.802
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
def load(path): return [json.loads(l) for l in open(path)]

# Fig A: monopoly check
fig,ax=plt.subplots(figsize=(8,3.6))
for r,c in zip([1,2,3],["#3b6fb6","#c0504d","#4f9a5b"]):
    L=load(f"results/monopoly_P0_run{r}.jsonl"); a=L[0]["alpha"]
    ax.plot([d["period"] for d in L],[d["agents"][0]["price"]/a for d in L],lw=1.3,color=c,label=f"α = {a:g}")
ax.axhline(P1M,ls="--",color="k",lw=1); ax.text(101,P1M,"monopoly\nprice 1.802",va="center",fontsize=8)
ax.axhspan(P1M*0.95,P1M*1.05,color="0.9",zorder=0)
ax.set_xlabel("Period"); ax.set_ylabel("Price / α"); ax.legend(frameon=False,loc="upper right"); ax.text(99,P1M*0.95-0.02,"shaded: ±5% of monopoly price",ha="right",va="top",fontsize=8)
ax.set_xlim(1,100); fig.tight_layout(); fig.savefig(OUT+"v2_monopoly_check.png",dpi=200); plt.close()

# Fig B: variance decomposition
V=pd.read_csv(R+"v2_variance_decomposition.csv",index_col=0)
fig,axes=plt.subplots(1,2,figsize=(11,3.6),gridspec_kw={"width_ratios":[1.1,1]})
s=V["share_pct"].dropna(); labels=["Prompt\n(between arms)","Chance\n(between runs, same prompt)","Over time\n(within runs)"]
b=axes[0].bar(labels,s.values,color=["#c0504d","#3b6fb6","0.6"])
for x,v in zip(b,s.values): axes[0].text(x.get_x()+x.get_width()/2,v+1.5,f"{v:.0f}%",ha="center")
axes[0].set_ylabel("Share of total price variation (%)"); axes[0].set_ylim(0,85); axes[0].set_title("All runs, periods 51–100",fontsize=10)
w=V.loc[ARMS,["between_runs_pct","within_runs_pct"]]
axes[1].bar(ARMS,w.between_runs_pct,color=[cols[a] for a in ARMS],label="between runs")
axes[1].bar(ARMS,w.within_runs_pct,bottom=w.between_runs_pct,color=[cols[a] for a in ARMS],alpha=0.35,label="within runs")
for i,a in enumerate(ARMS):
    axes[1].text(i,w.between_runs_pct[a]/2,f"{w.between_runs_pct[a]:.0f}%",ha="center",color="w")
    axes[1].text(i,w.between_runs_pct[a]+w.within_runs_pct[a]/2,f"{w.within_runs_pct[a]:.0f}%",ha="center")
axes[1].set_ylabel("Share within arm (%)"); axes[1].set_title("Between vs within runs, by arm\n(solid = between runs, light = within runs)",fontsize=10)
fig.tight_layout(); fig.savefig(OUT+"v2_variance_decomposition.png",dpi=200); plt.close()

# Fig C: profits and stability
P=pd.read_csv(R+"v2_profit_stability.csv"); U=pd.read_csv(R+"v2_runs.csv")
D=P.merge(U[["arm","run","sd"]],on=["arm","run"])
specs=[("profit","Mean profit per firm"),("sd","Within-run price SD"),("big_cuts","Sharp price cuts per run")]
fig,axes=plt.subplots(1,3,figsize=(12,3.7)); rng=np.random.default_rng(1)
for ax,(c,t) in zip(axes,specs):
    data=[D[D.arm==a][c].values for a in ARMS]
    bp=ax.boxplot(data,widths=0.5,showfliers=False,patch_artist=True,medianprops={"color":"k"})
    for p,a in zip(bp["boxes"],ARMS): p.set_facecolor(cols[a]); p.set_alpha(0.35)
    for i,d in enumerate(data): ax.plot(i+1+rng.uniform(-.12,.12,len(d)),d,"o",ms=3,color=cols[ARMS[i]])
    for i,d in enumerate(data): ax.plot(i+1,d.mean(),"D",ms=6,color="k",mfc="w",zorder=5)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(ARMS); ax.set_title(t,fontsize=10)
axes[0].axhline(22.29,ls=":",color="k",lw=1); axes[0].text(3.45,22.29,"Nash",fontsize=8,va="center")
fig.tight_layout(); fig.savefig(OUT+"v2_profit_stability.png",dpi=200); plt.close()

# Fig D: keywords
K=pd.read_csv(R+"v2_plans_keywords.csv").set_index("term")
terms=["explore/test","undercut","aggressive","higher price","stabil","long run/term"]
nice={"explore/test":"Explore / test","undercut":"Undercut","aggressive":"Aggressive","higher price":"Higher price","stabil":"Stability","long run/term":"Long run / long term"}
fig,ax=plt.subplots(figsize=(9,4)); y=np.arange(len(terms)); h=0.26
for j,a in enumerate(ARMS):
    ax.barh(y+(1-j)*h,K.loc[terms,a],height=h,color=cols[a],label=a)
for i,t in enumerate(terms):
    for j,a in enumerate(ARMS[1:]):
        p=K.loc[t,f"{a}_p"]
        if p<0.05: ax.text(K.loc[t,a]+1,i+(1-(j+1))*h,"*",va="center",fontsize=12,color=cols[a])
ax.set_yticks(y); ax.set_yticklabels([nice[t] for t in terms]); ax.invert_yaxis()
ax.set_xlabel("Share of agent-periods mentioning the term (%)"); ax.legend(frameon=False,loc="lower right")
ax.set_title("* = differs from P1 at the 5% level (Mann–Whitney on run-level shares)",fontsize=9)
fig.tight_layout(); fig.savefig(OUT+"v2_keywords.png",dpi=200); plt.close()

# Fig E: example runs (run closest to arm median of run means)
fig,axes=plt.subplots(1,3,figsize=(13,3.6),sharey=True)
for ax,a in zip(axes,ARMS):
    g=U[U.arm==a]; r=int(g.iloc[(g["mean"]-g["mean"].median()).abs().argmin()]["run"])
    L=load(f"{R}duopoly_{a}_run{r}.jsonl"); al=L[0]["alpha"]
    for k,ls in [(0,"-"),(1,"--")]:
        ax.plot([d["period"] for d in L],[d["agents"][k]["price"]/al for d in L],ls,lw=1.1,color=cols[a] if k==0 else "0.3",label=f"Firm {k+1}")
    ax.axhline(PN,color="k",lw=.8,ls=":"); ax.axhline(PM,color="k",lw=.8,ls=":")
    ax.text(100,PN,"Nash",fontsize=7,ha="right",va="bottom"); ax.text(100,PM,"Monopoly",fontsize=7,ha="right",va="bottom")
    ax.axvspan(51,100,color="0.94",zorder=0); ax.set_title(f"{a}, run {r} (α = {al:g})",fontsize=10); ax.set_xlabel("Period")
axes[0].set_ylabel("Price / α"); axes[0].legend(frameon=False,fontsize=8,loc="upper left")
fig.tight_layout(); fig.savefig(OUT+"v2_example_runs.png",dpi=200); plt.close()
print("ok")
