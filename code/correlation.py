import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    datapath = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../data/processed_data.csv'))

    df = pd.read_csv(datapath)
    score = df['score']
    good_score = df['score'] > 0
    is_answered = df['is_answered']
    drop = ['score', 'is_answered']
    df = df[df.columns.drop(drop)]

    score_correlations = {var: np.corrcoef(df[var], score)[0, 1] for var in df.columns}
    # print(score_correlations)
    keys = list(score_correlations.keys())
    vals = list(score_correlations.values())
    keys = [x for _, x in sorted(zip(vals, keys), key=lambda a: np.abs(a[0]))]
    vals.sort(key=np.abs)

    fig, ax = plt.subplots()
    ax.barh(keys, vals, height=0.2)
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    ax.grid(color='grey',
            linestyle='-.', linewidth=0.5,
            alpha = 0.2)
    ax.set_xlabel('Pearson Correlation Coefficient')
    ax.set_ylabel('Features')
    plt.subplots_adjust(left=0.25)
    # plt.show()
    # savedir = os.path.abspath(os.path.join(os.path.realpath(__file__), 
    #                     '../../deliverables/preliminary_results/correlation.png'))
    savedir = os.path.abspath(os.path.join(os.path.realpath(__file__), 
                        '../../report/figures/correlation.png'))
    plt.savefig(savedir, bbox_inches='tight')
