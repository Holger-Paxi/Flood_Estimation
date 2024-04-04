# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# %% [markdown]
# # inputs

# %%
input_excel_filepath = './Input_FE_Assignment_1/Powells Creek AMS.xlsx'
input_sheet_name = 'Sheet1'
output_dir = './Output_FE_Assignment_1__1st/'
# case 1: 11-year data
input_1st_sample = [1960, 1971]
input_2nd_sample = [1970, 1981]
input_3rd_sample = [1980, 1991]
# case 2: 10-year data - forward
input_1st_sample_mod1 = [1960, 1970]
input_2nd_sample_mod1 = [1970, 1980]
input_3rd_sample_mod1 = [1980, 1990]
# case 3: 10-year data - backward
input_1st_sample_mod2 = [1961, 1971]
input_2nd_sample_mod2 = [1971, 1981]
input_3rd_sample_mod2 = [1981, 1991]

print(
    input_excel_filepath,
    input_sheet_name,
    output_dir,
    # case 1: 11-year data
    input_1st_sample,
    input_2nd_sample,
    input_3rd_sample,
    # case 2: 10-year data - forward
    input_1st_sample_mod1,
    input_2nd_sample_mod1,
    input_3rd_sample_mod1,
    # case 3: 10-year data - backward
    input_1st_sample_mod2,
    input_2nd_sample_mod2,
    input_3rd_sample_mod2,
    sep='\n'
    )

# %%
case_1 = [input_1st_sample, input_2nd_sample, input_3rd_sample,]
case_2 = [input_1st_sample_mod1, input_2nd_sample_mod1, input_3rd_sample_mod1]
case_3 = [input_1st_sample_mod2, input_2nd_sample_mod2, input_3rd_sample_mod2]

print(
    case_1,
    case_2,
    case_3,
    sep='\n'
    )

# %%
def create_output_dir(arg_output_dir):
    """create output directory if it does not exist

    arguments:
        [string] --> arg_output_dir = path of the output directory name
    """
    if not os.path.exists(arg_output_dir):
        os.makedirs(arg_output_dir)

# %%
create_output_dir(output_dir)
output_dir

# %% [markdown]
# # full data

# %%
data = pd.read_excel(io=input_excel_filepath, sheet_name=input_sheet_name, header=None, names=['year', 'flow_rate'])
data.to_csv(path_or_buf='{}full_data.csv'.format(output_dir), header=False, index=False)
data

# %%
fig, ax = plt.subplots(figsize=(10,4))

ax.plot(
    data.year.to_numpy(), 
    data.flow_rate.to_numpy(),
    '-o'
    )

ax.grid(visible=True)
ax.set_title(label='Time Series for the {}-year sample'.format(len(data)))
ax.set_xlabel(xlabel='Time in ($year$)')
ax.set_ylabel(ylabel='Flow Rate in ($m^3/s$)')
fig.savefig(fname='{}time_series_40_year_sample.png'.format(output_dir))

# %% [markdown]
# # data case 1

# %%
data_c1 = [data.iloc[data[data.year == ind[0]].index[0]:data[data.year == ind[1]].index[0]] for ind in case_1]
ind_ini = [ind[0] for ind in case_1]
ind_fin = [ind[-1]-1 for ind in case_1]
ind_enu = range(len(data_c1))
[ind1.to_csv(
    path_or_buf='{}data_case_1_{}_{}_{}.csv'.format(output_dir, ind2, ind3, ind4),
    header=False,
    index=False
    ) for ind1, ind2, ind3, ind4 in zip(data_c1, ind_enu, ind_ini, ind_fin)]

data_c1

# %%
for ind1, ind2, ind3 in zip(data_c1, ind_ini, ind_fin):
    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        ind1.year.to_numpy(), 
        ind1.flow_rate.to_numpy(),
        '-o'
        )

    ax.grid(visible=True)
    ax.set_title(label='Time Series for the {}-year sample\nfrom {} to {}'.format(
        len(ind1),
        ind2, 
        ind3
        ))
    ax.set_xlabel(xlabel='Time in ($year$)')
    ax.set_ylabel(ylabel='Flow Rate in ($m^3/s$)')
    fig.savefig(fname='{}time_series_case_1_{}_year_sample_{}_{}.png'.format(
        output_dir, 
        len(ind1), 
        ind2, 
        ind3
        ))

# %%
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,12))

ax[0].plot(
    data_c1[0].year.to_numpy(), 
    data_c1[0].flow_rate.to_numpy(),
    '-o'
    )

ax[0].grid(visible=True)
ax[0].set_title(
    label='{}-year sample from {} to {}'.format(len(data_c1[0]), ind_ini[0], ind_fin[0]),
    loc='left'
    )

ax[1].plot(
    data_c1[1].year.to_numpy(), 
    data_c1[1].flow_rate.to_numpy(),
    '-o'
    )

ax[1].grid(visible=True)
ax[1].set_title(
    label='{}-year sample from {} to {}'.format(len(data_c1[1]), ind_ini[1], ind_fin[1]),
    loc='left'
    )

ax[2].plot(
    data_c1[2].year.to_numpy(), 
    data_c1[2].flow_rate.to_numpy(),
    '-o'
    )

ax[2].grid(visible=True)
ax[2].set_title(
    label='{}-year sample from {} to {}'.format(len(data_c1[2]), ind_ini[2], ind_fin[2]),
    loc='left'
    )

fig.suptitle(t='Time Series - Case 1', x=0.5, y=0.92)
fig.supxlabel(t='Time in ($year$)', x=0.5, y=0.07)
fig.supylabel(t='Flow Rate in ($m^3/s$)', x=0.07, y=0.5)

fig.savefig(fname='{}time_series_case_1.png'.format(
    output_dir, 
    ))

# %%
data_c1 = pd.concat(objs=[ind.reset_index(drop=True) for ind in data_c1], axis=1)
_ = []
for ind1, ind2 in zip(ind_ini, ind_fin):
    _.append('{}_{}'.format(ind1, ind2))
    _.append('{}_{}'.format(ind1, ind2))
data_c1.columns = pd.MultiIndex.from_tuples(tuples=[(ind1, ind2) for ind1, ind2 in zip(_, data_c1.columns)])
data_c1.to_csv(path_or_buf='{}data_case_1.csv'.format(output_dir), header=True, index=True)

data_c1

# %% [markdown]
# # data case 2

# %%
data_c2 = [data.iloc[data[data.year == ind[0]].index[0]:data[data.year == ind[1]].index[0]] for ind in case_2]
ind_ini = [ind[0] for ind in case_2]
ind_fin = [ind[-1]-1 for ind in case_2]
ind_enu = range(len(data_c2))
[ind1.to_csv(
    path_or_buf='{}data_case_2_{}_{}_{}.csv'.format(output_dir, ind2, ind3, ind4),
    header=False,
    index=False
    ) for ind1, ind2, ind3, ind4 in zip(data_c2, ind_enu, ind_ini, ind_fin)]

data_c2

# %%
for ind1, ind2, ind3 in zip(data_c2, ind_ini, ind_fin):
    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        ind1.year.to_numpy(), 
        ind1.flow_rate.to_numpy(),
        '-o'
        )

    ax.grid(visible=True)
    ax.set_title(label='Time Series for the {}-year sample\nfrom {} to {}'.format(
        len(ind1),
        ind2, 
        ind3
        ))
    ax.set_xlabel(xlabel='Time in ($year$)')
    ax.set_ylabel(ylabel='Flow Rate in ($m^3/s$)')
    fig.savefig(fname='{}time_series_case_2_{}_year_sample_{}_{}.png'.format(
        output_dir, 
        len(ind1), 
        ind2, 
        ind3
        ))

# %%
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,12))

ax[0].plot(
    data_c2[0].year.to_numpy(), 
    data_c2[0].flow_rate.to_numpy(),
    '-o'
    )

ax[0].grid(visible=True)
ax[0].set_title(
    label='{}-year sample from {} to {}'.format(len(data_c2[0]), ind_ini[0], ind_fin[0]),
    loc='left'
    )

ax[1].plot(
    data_c2[1].year.to_numpy(), 
    data_c2[1].flow_rate.to_numpy(),
    '-o'
    )

ax[1].grid(visible=True)
ax[1].set_title(
    label='{}-year sample from {} to {}'.format(len(data_c2[1]), ind_ini[1], ind_fin[1]),
    loc='left'
    )

ax[2].plot(
    data_c2[2].year.to_numpy(), 
    data_c2[2].flow_rate.to_numpy(),
    '-o'
    )

ax[2].grid(visible=True)
ax[2].set_title(
    label='{}-year sample from {} to {}'.format(len(data_c2[2]), ind_ini[2], ind_fin[2]),
    loc='left'
    )

fig.suptitle(t='Time Series - Case 2', x=0.5, y=0.92)
fig.supxlabel(t='Time in ($year$)', x=0.5, y=0.07)
fig.supylabel(t='Flow Rate in ($m^3/s$)', x=0.07, y=0.5)

fig.savefig(fname='{}time_series_case_2.png'.format(
    output_dir, 
    ))

# %%
data_c2 = pd.concat(objs=[ind.reset_index(drop=True) for ind in data_c2], axis=1)
_ = []
for ind1, ind2 in zip(ind_ini, ind_fin):
    _.append('{}_{}'.format(ind1, ind2))
    _.append('{}_{}'.format(ind1, ind2))
data_c2.columns = pd.MultiIndex.from_tuples(tuples=[(ind1, ind2) for ind1, ind2 in zip(_, data_c2.columns)])
data_c2.to_csv(path_or_buf='{}data_case_2.csv'.format(output_dir), header=True, index=True)

data_c2

# %% [markdown]
# # data case 3

# %%
data_c3 = [data.iloc[data[data.year == ind[0]].index[0]:data[data.year == ind[1]].index[0]] for ind in case_3]
ind_ini = [ind[0] for ind in case_3]
ind_fin = [ind[-1]-1 for ind in case_3]
ind_enu = range(len(data_c3))
[ind1.to_csv(
    path_or_buf='{}data_case_3_{}_{}_{}.csv'.format(output_dir, ind2, ind3, ind4),
    header=False,
    index=False
    ) for ind1, ind2, ind3, ind4 in zip(data_c3, ind_enu, ind_ini, ind_fin)]

data_c3

# %%
for ind1, ind2, ind3 in zip(data_c3, ind_ini, ind_fin):
    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        ind1.year.to_numpy(), 
        ind1.flow_rate.to_numpy(),
        '-o'
        )

    ax.grid(visible=True)
    ax.set_title(label='Time Series for the {}-year sample\nfrom {} to {}'.format(
        len(ind1),
        ind2, 
        ind3
        ))
    ax.set_xlabel(xlabel='Time in ($year$)')
    ax.set_ylabel(ylabel='Flow Rate in ($m^3/s$)')
    fig.savefig(fname='{}time_series_case_3_{}_year_sample_{}_{}.png'.format(
        output_dir, 
        len(ind1), 
        ind2, 
        ind3
        ))

# %%
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,12))

ax[0].plot(
    data_c3[0].year.to_numpy(), 
    data_c3[0].flow_rate.to_numpy(),
    '-o'
    )

ax[0].grid(visible=True)
ax[0].set_title(
    label='{}-year sample from {} to {}'.format(len(data_c3[0]), ind_ini[0], ind_fin[0]),
    loc='left'
    )

ax[1].plot(
    data_c3[1].year.to_numpy(), 
    data_c3[1].flow_rate.to_numpy(),
    '-o'
    )

ax[1].grid(visible=True)
ax[1].set_title(
    label='{}-year sample from {} to {}'.format(len(data_c3[1]), ind_ini[1], ind_fin[1]),
    loc='left'
    )

ax[2].plot(
    data_c3[2].year.to_numpy(), 
    data_c3[2].flow_rate.to_numpy(),
    '-o'
    )

ax[2].grid(visible=True)
ax[2].set_title(
    label='{}-year sample from {} to {}'.format(len(data_c3[2]), ind_ini[2], ind_fin[2]),
    loc='left'
    )

fig.suptitle(t='Time Series - Case 3', x=0.5, y=0.92)
fig.supxlabel(t='Time in ($year$)', x=0.5, y=0.07)
fig.supylabel(t='Flow Rate in ($m^3/s$)', x=0.07, y=0.5)

fig.savefig(fname='{}time_series_case_3.png'.format(
    output_dir, 
    ))

# %%
data_c3 = pd.concat(objs=[ind.reset_index(drop=True) for ind in data_c3], axis=1)
_ = []
for ind1, ind2 in zip(ind_ini, ind_fin):
    _.append('{}_{}'.format(ind1, ind2))
    _.append('{}_{}'.format(ind1, ind2))
data_c3.columns = pd.MultiIndex.from_tuples(tuples=[(ind1, ind2) for ind1, ind2 in zip(_, data_c3.columns)])
data_c3.to_csv(path_or_buf='{}data_case_3.csv'.format(output_dir), header=True, index=True)

data_c3

# %%



