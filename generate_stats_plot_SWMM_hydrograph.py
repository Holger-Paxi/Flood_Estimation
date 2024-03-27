# %% [markdown]
# # import python packages

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import seaborn as sns

# %% [markdown]
# # list parquet files

# %%
parquet_files = []

for file in os.listdir('../Output_SWMM_directory/'):
    if file.endswith('.parquet'):
        parquet_files.append(file)

parquet_files = pd.DataFrame(parquet_files, columns=['parquet_filename'])
parquet_files.sort_values(by='parquet_filename', inplace=True, ignore_index=True)

# %% [markdown]
# # read parquet files

# %%
hydrograph_list = []

for index_1 in parquet_files.index:
    a1 = parquet_files.parquet_filename[index_1]
    a2 = pd.read_parquet('../Output_SWMM_directory/' + a1)
    a3 = pd.DataFrame({'frequency':[a1.split('__',1)[0]], 
                       'freq_AEP':[a1[14:].split('__',1)[0]], 
                       'duration':[int(a1[:-8][-7:-4])], 
                       'pattern':[int(a1[:-8][-2:])], 
                       'hydrograph':[a2]})
    hydrograph_list.append(a3)

hydrograph_list = pd.concat(hydrograph_list, ignore_index=True)

# %% [markdown]
# # get peak flow for each hydrograph

# %%
max_flow_list = []

for index_2 in hydrograph_list.index:
    b1 = hydrograph_list.hydrograph[index_2].flow.max()
    max_flow_list.append(b1)

max_flow_list = pd.DataFrame({'max_flow':max_flow_list})
max_flow_list = pd.concat([hydrograph_list.iloc[:,:4], max_flow_list], axis=1)
max_flow_list.to_csv('../GSPSWMMH_max_flows.csv', index=False)

# %% [markdown]
# # statistics

# %%
boundary_limits = pd.DataFrame(max_flow_list.groupby(by=max_flow_list.columns[:3].to_list()).max_flow.quantile(q=[0.05, 0.95])).unstack(level=-1)
boundary_limits.columns = [[ind[0] for ind in boundary_limits.columns], ['{}%'.format(int(ind[1] * 100)) for ind in boundary_limits.columns]]

# %%
statistics = max_flow_list.groupby(by=max_flow_list.columns[:3].to_list()).agg({'max_flow':['describe']})
statistics.columns = [[ind[0] for ind in statistics.columns], [ind[2] for ind in statistics.columns]]
statistics = pd.concat([statistics, boundary_limits], axis=1)
statistics = statistics[statistics.columns[[0, 1, 2, 3, 8, 4, 5, 6, 9, 7]]]
statistics.to_csv('../GSPSWMMH_statistics.csv')

# %%
design_flows = statistics[statistics.columns[[1, 4, 6, 8]]]
design_flows.reset_index(inplace=True)
design_flows.columns = [ind[0] if ind[1] == '' else ind[1] for ind in design_flows.columns]
temp_var = list(design_flows.groupby(by=list(design_flows.columns[:2])))
design_flows = pd.concat(list(val[1].loc[val[1][val[1]['50%'] == val[1]['50%'].max()].index] 
                              for ind, val in enumerate(temp_var)), ignore_index=True)
design_flows = pd.concat([design_flows, pd.DataFrame({'AEP':[63.21, 50, 39.35, 20, 18.13, 10, 5, 2, 1]})], axis=1)
design_flows = design_flows[design_flows.columns[[0, 1, 2, 7, 3, 4, 5, 6]]]
design_flows.to_csv('../GSPSWMMH_design_flows.csv', index=False)

# %% [markdown]
# # plot

# %%
titles = max_flow_list[['frequency', 'freq_AEP']].drop_duplicates(ignore_index=True)

# %%
for index_3 in titles.index:
    c1 = max_flow_list[(max_flow_list.frequency == titles.frequency[index_3]) & (max_flow_list.freq_AEP == titles.freq_AEP[index_3])]
    c2 = statistics.loc[titles.frequency[index_3], titles.freq_AEP[index_3]].T
    c3 = plt.figure(figsize=(15,5))
    c4 = sns.boxplot(data=c1,
                     x='duration',
                     y='max_flow',
                     )
    c4.set(title='{} {}'.format(titles.frequency[index_3], titles.freq_AEP[index_3][7:]))
    c5 = plt.table(cellText=c2.to_numpy().round(decimals=3),
                   rowLabels=list(ind[1] for ind in c2.index),
                   colLabels=list(c2.columns),
                   bbox=[0, -0.65, 1, 0.5],
                   )
    c5.auto_set_font_size(False)
    c5.set_fontsize(10)
    c6 = plt.savefig('../GSPSWMMH__{}__{}.png'.format(titles.frequency[index_3], titles.freq_AEP[index_3][7:]), bbox_inches='tight')

# %%
d0 = plt.figure(figsize=(15,6))
d1 = sns.lineplot(data=design_flows[design_flows.columns[3:]].set_index('AEP'))
d1.set(xscale='log')
d1.invert_xaxis()
d1.grid()
d1.scatter(data=design_flows, 
           x='AEP',
           y='50%'
           )
d0 = plt.table(cellText=design_flows[['AEP', 'duration', 'mean', '5%', '50%', '95%']].T.to_numpy().round(decimals=3),
               rowLabels=list(design_flows[['AEP', 'duration', 'mean', '5%', '50%', '95%']].T.index),
               bbox=[0, -0.45, 1, 0.3],
               )
line = d1.get_lines()
d0 = plt.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[3].get_ydata(), color='blue', alpha=.15)
d2 = plt.savefig('../GSPSWMMH_design_flows.png', bbox_inches='tight')
