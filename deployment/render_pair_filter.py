"""Export concise tables and figures from a completed historical pair audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def render(work):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work=Path(work)
    assert json.loads((work/'summary.json').read_text())['complete'], 'Full cohort is not complete'
    pooled=pd.read_csv(work/'cohort_comparison.csv')
    names={'chunkedgraph':'L2 paths / L2 endpoints',
           'skeleton_matched_l2':'Skeleton paths / L2 endpoints',
           'skeleton_native_nodes':'Skeleton paths / native endpoints'}
    policies=['all_pairs','uniform_distance']
    policy_names={'all_pairs':'All pairs','uniform_distance':'Uniform distance'}
    compact=[]
    for mode in names:
        for policy in policies:
            rows=pooled[(pooled['mode']==mode)&(pooled.policy==policy)]
            assert len(rows)==5 and rows.neurons.nunique()==1
            np.testing.assert_allclose(rows.baseline_error_percent,rows.baseline_error_percent.iloc[0],atol=1e-10)
            compact.append(dict(geometry=names[mode],policy=policy_names[policy],neurons=int(rows.neurons.iloc[0]),
                before_percent=float(rows.baseline_error_percent.iloc[0]),
                after_min_percent=float(rows.error_percent.min()),after_max_percent=float(rows.error_percent.max()),
                retained_min_percent=float(rows.retained_weight_percent.min()),retained_max_percent=float(rows.retained_weight_percent.max()),
                improved_min_neurons=int(rows.neurons_with_lower_error.min()),improved_max_neurons=int(rows.neurons_with_lower_error.max()),
                worsened_min_neurons=int(rows.neurons_with_higher_error.min()),worsened_max_neurons=int(rows.neurons_with_higher_error.max())))
    pd.DataFrame(compact).to_csv(work/'comparison_compact.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
    colors=plt.cm.viridis(np.linspace(.1,.85,5))
    for ax,policy in zip(axes,policies):
        for m,mode in enumerate(names):
            r=pooled[(pooled['mode']==mode)&(pooled.policy==policy)].sort_values('fold')
            baseline=r.baseline_error_percent.iloc[0]
            ax.scatter(m-.2,baseline,marker='s',color='black',s=45,label='Before filtering' if m==0 else None,zorder=4)
            for fold,row in enumerate(r.itertuples()):
                x=m+.02+fold*.065
                ax.plot([m-.2,x],[baseline,row.error_percent],color=colors[fold],alpha=.35,lw=1)
                ax.scatter(x,row.error_percent,color=colors[fold],s=30,label=f'Fold {fold}' if m==0 else None,zorder=4)
        ax.set_xticks(range(3),['L2','Skeleton\nmatched L2','Skeleton\nnative'])
        ax.set_title(policy_names[policy]);ax.grid(axis='y',alpha=.2);ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Pooled incorrect pair weight (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Same-subcompartment filter · 1,924 neurons · 150 µm cutoff')
    for suffix in ['png','pdf']:fig.savefig(work/f'pooled_comparison.{suffix}',dpi=180)
    plt.close(fig)
    cells=pd.read_csv(work/'cell_scores.csv',dtype={'root_1718':str})
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained',sharex=True,sharey=True)
    for i,policy in enumerate(policies):
        for j,mode in enumerate(names):
            ax=axes[i,j];r=cells[(cells['mode']==mode)&(cells.policy==policy)&(cells.fold==0)]
            ax.scatter(r.baseline_error_percent,r.error_percent,s=5,alpha=.3,rasterized=True)
            ax.plot([0,100],[0,100],color='black',lw=.8,ls='--')
            ax.set(xlim=(0,100),ylim=(0,100),aspect='equal')
            ax.set_title(names[mode]+'\n'+policy_names[policy],fontsize=10)
            if j==0:ax.set_ylabel('After filtering (%)')
            if i==1:ax.set_xlabel('Before filtering (%)')
    fig.suptitle('Per-neuron pair error · classifier fold 0 · below diagonal = improvement')
    for suffix in ['png','pdf']:fig.savefig(work/f'neuron_comparison.{suffix}',dpi=180)
    plt.close(fig)
    print(pd.DataFrame(compact).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',default='/orcd/scratch/orcd/013/jcbliao/segclr_pair_audit/v117_subcompartment_filter_20261108')
    render(p.parse_args().work)
