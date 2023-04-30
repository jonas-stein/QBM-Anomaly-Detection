#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

stats = pd.read_csv("hidden_metrics.csv")
categories = ['f1_score']

stats_by_approach = stats.groupby(['approach'])
approach_names = stats_by_approach.groups.keys()

ax = plt.gca()

for approach in approach_names:
    data = stats_by_approach.get_group(approach)
    max_total = float(data['f1_score'].max())
    max_total = math.ceil(max_total)

    data = data.sort_values(by=['hidden_units'])
    print(data.sort_values(by=['f1_score'], ascending=False).iloc[0])


    data.plot(ax=ax,
              x='hidden_units',
              y=[cat for cat in categories],
              label=[f'{approach}_{categories[i]}' for i in range(len(categories))],
              #ylim=(0, max_total),
              #yticks=np.arange(0, 1, 0.05),
              #xticks=range(1, stats['hidden_units'].max()+2, 1),
              ylabel='F1-Score',
              xlabel='Batchsize')

plt.grid()
plt.savefig('batchsize_figure.pdf', bbox_inches='tight', pad_inches=0.2)
