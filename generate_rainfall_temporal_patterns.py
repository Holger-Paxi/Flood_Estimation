# %% [markdown]
# # import python packages

# %%
import pandas as pd
import datetime as dt
import shutil
import os

# %% [markdown]
# # AEP frequencies

# %%
AEP_frequency = pd.DataFrame({'frequency':['frequent', 'frequent', 'frequent', 'frequent', 'frequent', 'intermediate', 'intermediate', 'rare', 'rare'], 
                              'EY':[1, 0.69, 0.5, 0.22, 0.2, 0.11, 0.05, 0.02, 0.01],
                              'AEP_in_percentage':[63.21, 50, 39.35, 20, 18.13, 10, 5, 2, 1], 
                              'AEP_1_in_X_years':[1.58, 2, 2.54, 5, 5.52, 10, 20, 50, 100],
                              'ARI':[1, 1.44, 2, 4.48, 5, 9.49, 19.5, 49.5, 99.5]})

# %% [markdown]
# # read inputs

# %% [markdown]
# ## precipitation burst duration

# %%
df_statistics = pd.read_csv('../Inputs_FE_Assignment_3/ECsouth/ECsouth_AllStats.csv')
df_statistics = df_statistics.dropna(axis=0)
column_statistics = [df_statistics.columns.to_list()[index_1].replace(' ', '_') for index_1 in range(len(df_statistics.columns.to_list()))]
column_statistics = [column_statistics[index_1].replace('(', '') for index_1 in range(len(column_statistics))]
column_statistics = [column_statistics[index_1].replace(')', '') for index_1 in range(len(column_statistics))]
column_statistics = [column_statistics[index_1].replace('.', '') for index_1 in range(len(column_statistics))]
column_statistics = [column_statistics[index_1].replace('%', 'percentage') for index_1 in range(len(column_statistics))]
df_statistics.rename(columns={df_statistics.columns.to_list()[index_1]: column_statistics[index_1] for index_1 in range(len(column_statistics))}, inplace=True)

# %% [markdown]
# ## rainfall patterns

# %%
df_increments = pd.read_csv('../Inputs_FE_Assignment_3/ECsouth/ECsouth_Increments.csv')
df_increments.dropna(axis=0)
column_increments = [df_increments.columns.to_list()[index_2].replace(' ', '_') for index_2 in range(len(df_increments.columns.to_list()))]
column_increments = [column_increments[index_2].replace(':', '') for index_2 in range(len(column_increments))]
df_increments.rename(columns={df_increments.columns.to_list()[index_2]: column_increments[index_2] for index_2 in range(len(column_increments))}, inplace=True)

# %% [markdown]
# ## precipitation depth

# %%
df_prec_depths = pd.read_csv('../Inputs_FE_Assignment_3/depths_33.8774_151.093/depths_-33.8774_151.093_all_design.csv', skiprows=9)
column_prec_depths = [df_prec_depths.columns.to_list()[index_3].replace(' ', '_') for index_3 in range(len(df_prec_depths.columns.to_list()))]
column_prec_depths = [column_prec_depths[index_3].replace('%', '_percentage') for index_3 in range(len(column_prec_depths))]
column_prec_depths = [column_prec_depths[index_3].replace('EY', '_EY') for index_3 in range(len(column_prec_depths))]
column_prec_depths = [column_prec_depths[index_3].replace('.', '_') for index_3 in range(len(column_prec_depths))]
df_prec_depths.rename(columns={df_prec_depths.columns.to_list()[index_3]: 
                               column_prec_depths[index_3] for index_3 in range(len(column_prec_depths[:2]))}, 
                               inplace=True)
df_prec_depths.rename(columns={df_prec_depths.columns.to_list()[2:][index_3]: 
                               'freq{:>02}_'.format(index_3) + column_prec_depths[2:][index_3] for index_3 in range(len(column_prec_depths[2:]))}, 
                               inplace=True)

# %% [markdown]
# # data required

# %%
timestep = 5

# %%
starting_date = dt.datetime(year=2023, month=1, day=1, hour=0, minute=5, second=0)

# %%
rain_gauge = 'RainSeries1'

# %%
orig_SWMM_file = '../Inputs_FE_Assignment_3/971007_SW5.inp'

# %%
output_directory = 'Output_SWMM_directory'

# %%
time_delta = dt.timedelta(minutes=timestep)

# %%
max_duration = df_increments[df_increments._TimeStep == timestep]._Duration.max()

# %%
min_duration = df_increments[df_increments._TimeStep == timestep]._Duration.min()

# %%
AEP_frequency_req = pd.concat([AEP_frequency, pd.DataFrame({'frequency_descriptor':df_prec_depths.columns.to_list()[7:16]})], axis=1)

# %%
max_len_freq = AEP_frequency_req.frequency.apply(lambda arg: len(arg)).max()

# %%
max_len_desc = AEP_frequency_req.frequency_descriptor.apply(lambda arg: len(arg)).max()

# %% [markdown]
# ## burst duration

# %%
df_stats = df_statistics[['Burst_Duration_min', 'Burst_Loading', 'AEP_Window']]
df_stats = df_stats[df_stats.Burst_Duration_min <= max_duration]
df_stats.reset_index(drop=True, inplace=True)

# %%
grouping_by_duration_stats = df_stats.Burst_Duration_min.drop_duplicates().reset_index(drop=True)

# %%
grouping_by_AEP_stats = df_stats.AEP_Window.drop_duplicates().reset_index(drop=True)

# %%
list_stats = []

for index_4 in grouping_by_duration_stats.index:
    a1 = df_stats[df_stats.Burst_Duration_min == grouping_by_duration_stats[index_4]].dropna(axis=1)
    for index_5 in grouping_by_AEP_stats.index:
        a2 = a1[a1.AEP_Window == grouping_by_AEP_stats[index_5]]
        a2.reset_index(drop=True, inplace=True)
        a3 = pd.DataFrame({'Duration': [grouping_by_duration_stats[index_4]], 'AEP': [grouping_by_AEP_stats[index_5]], 'burst_duration': [a2]})
        list_stats.append(a3)

list_stats = pd.concat(list_stats, ignore_index=True)

# %% [markdown]
# ## temporal patterns

# %%
df_incre = df_increments.drop(columns=['EventID', '_Region'], axis=1)
df_incre.rename(columns={'_Duration':'Duration', '_TimeStep':'TimeStep', '_AEP':'AEP', '_Increments':'Increments'}, inplace=True)
df_incre = df_incre[(df_incre.TimeStep == timestep)]
df_incre.reset_index(drop=True, inplace=True)
df_incre.rename(columns={df_incre.columns[3:][index_6]: 'inc_{}'.format(index_6) for index_6 in range(len(df_incre.columns[3:]))}, inplace=True)
df_incre.drop(columns=df_incre.columns[df_incre.sum(axis=0) == 0], inplace=True)

# %%
grouping_by_duration_incre = df_incre.Duration.drop_duplicates().reset_index(drop=True)

# %%
grouping_by_AEP_incre = df_incre.AEP.drop_duplicates().reset_index(drop=True)

# %%
list_incre = []

for index_7 in grouping_by_duration_incre.index:
    b1 = df_incre[df_incre.Duration == grouping_by_duration_incre[index_7]].dropna(axis=1)
    for index_8 in grouping_by_AEP_incre.index:
        b2 = b1[b1.AEP == grouping_by_AEP_incre[index_8]]
        b2.reset_index(drop=True, inplace=True)
        b3 = pd.DataFrame({'Duration': [grouping_by_duration_incre[index_7]], 'AEP': [grouping_by_AEP_incre[index_8]], 'increments': [b2]})
        list_incre.append(b3)

list_incre = pd.concat(list_incre, ignore_index=True)

# %% [markdown]
# ## precipitation depth

# %%
df_depth = df_prec_depths[['Duration_in_min'] + AEP_frequency_req.frequency_descriptor.to_list()]
df_depth = df_depth[(df_depth.Duration_in_min >= min_duration) & (df_depth.Duration_in_min <= max_duration)]
df_depth.reset_index(drop=True, inplace=True)

# %% [markdown]
# # temporal patterns

# %%
list_temp_patterns = []

for index_9 in list_stats.index:
    c1 = list_stats.Duration[index_9]
    c2 = list_stats.AEP[index_9]
    c3 = pd.concat([list_stats.burst_duration[index_9].iloc[:,:2], list_incre.increments[index_9].iloc[:,1:]], axis=1)
    c4 = pd.DataFrame({'Duration':[c1], 'AEP':[c2], 'temp_patterns':[c3]})
    list_temp_patterns.append(c4)

list_temp_patterns = pd.concat(list_temp_patterns, ignore_index=True)
list_temp_patterns.sort_values(by=['AEP', 'Duration'], inplace=True, ignore_index=True)

# %% [markdown]
# # frequency

# %% [markdown]
# ## frequent

# %%
freq_temp_patterns = pd.concat([df_depth[['Duration_in_min'] + AEP_frequency_req[AEP_frequency_req.frequency == 'frequent'].frequency_descriptor.to_list()], 
                                list_temp_patterns[list_temp_patterns.AEP == 'frequent'].reset_index(drop=True)], axis=1)

# %%
freq_hyetograph_depths = []

for index_10 in freq_temp_patterns.index:
    list_temp = []
    for index_11 in AEP_frequency_req[AEP_frequency_req.frequency == 'frequent'].frequency_descriptor.index:
        d1 = freq_temp_patterns[AEP_frequency_req[AEP_frequency_req.frequency == 'frequent'].frequency_descriptor[index_11]][index_10]
        d2 = freq_temp_patterns.temp_patterns[index_10].iloc[:,4:]
        d3 = d1 * d2 / 100
        d4 = pd.DataFrame({AEP_frequency_req[AEP_frequency_req.frequency == 'frequent'].frequency_descriptor[index_11]:[d3]})
        list_temp.append(d4)
    list_temp = pd.concat(list_temp, axis=1)
    freq_hyetograph_depths.append(list_temp)

freq_hyetograph_depths = pd.concat(freq_hyetograph_depths, ignore_index=True)

# %%
freq_hyetograph_timestamp = []

for index_12 in freq_temp_patterns.index:
    list_temp = []
    e1 = freq_temp_patterns.temp_patterns[index_12].columns[4:]
    for index_13 in range(int(freq_temp_patterns.Duration[index_12] / timestep)):
        e2 = starting_date + index_13 * time_delta
        e3 = pd.DataFrame({e1[index_13]: [e2]})
        list_temp.append(e3)
    list_temp = pd.concat(list_temp, axis=1)
    e4 = pd.DataFrame({'increments': [list_temp]})
    freq_hyetograph_timestamp.append(e4)

freq_hyetograph_timestamp = pd.concat(freq_hyetograph_timestamp, ignore_index=True)

# %%
freq_rainseries = pd.concat([freq_temp_patterns.Duration_in_min, freq_temp_patterns.AEP, freq_hyetograph_timestamp, freq_hyetograph_depths], axis=1)

# %% [markdown]
# ## intermediate

# %%
inte_temp_patterns = pd.concat([df_depth[['Duration_in_min'] + AEP_frequency_req[AEP_frequency_req.frequency == 'intermediate'].frequency_descriptor.to_list()], 
                                list_temp_patterns[list_temp_patterns.AEP == 'intermediate'].reset_index(drop=True)], axis=1)

# %%
inte_hyetograph_depths = []

for index_10 in inte_temp_patterns.index:
    list_temp = []
    for index_11 in AEP_frequency_req[AEP_frequency_req.frequency == 'intermediate'].frequency_descriptor.index:
        d1 = inte_temp_patterns[AEP_frequency_req[AEP_frequency_req.frequency == 'intermediate'].frequency_descriptor[index_11]][index_10]
        d2 = inte_temp_patterns.temp_patterns[index_10].iloc[:,4:]
        d3 = d1 * d2 / 100
        d4 = pd.DataFrame({AEP_frequency_req[AEP_frequency_req.frequency == 'intermediate'].frequency_descriptor[index_11]:[d3]})
        list_temp.append(d4)
    list_temp = pd.concat(list_temp, axis=1)
    inte_hyetograph_depths.append(list_temp)

inte_hyetograph_depths = pd.concat(inte_hyetograph_depths, ignore_index=True)

# %%
inte_hyetograph_timestamp = []

for index_12 in inte_temp_patterns.index:
    list_temp = []
    e1 = inte_temp_patterns.temp_patterns[index_12].columns[4:]
    for index_13 in range(int(inte_temp_patterns.Duration[index_12] / timestep)):
        e2 = starting_date + index_13 * time_delta
        e3 = pd.DataFrame({e1[index_13]: [e2]})
        list_temp.append(e3)
    list_temp = pd.concat(list_temp, axis=1)
    e4 = pd.DataFrame({'increments': [list_temp]})
    inte_hyetograph_timestamp.append(e4)

inte_hyetograph_timestamp = pd.concat(inte_hyetograph_timestamp, ignore_index=True)

# %%
inte_rainseries = pd.concat([inte_temp_patterns.Duration_in_min, inte_temp_patterns.AEP, inte_hyetograph_timestamp, inte_hyetograph_depths], axis=1)

# %% [markdown]
# ## rare

# %%
rare_temp_patterns = pd.concat([df_depth[['Duration_in_min'] + AEP_frequency_req[AEP_frequency_req.frequency == 'rare'].frequency_descriptor.to_list()], 
                                list_temp_patterns[list_temp_patterns.AEP == 'rare'].reset_index(drop=True)], axis=1)

# %%
rare_hyetograph_depths = []

for index_10 in rare_temp_patterns.index:
    list_temp = []
    for index_11 in AEP_frequency_req[AEP_frequency_req.frequency == 'rare'].frequency_descriptor.index:
        d1 = rare_temp_patterns[AEP_frequency_req[AEP_frequency_req.frequency == 'rare'].frequency_descriptor[index_11]][index_10]
        d2 = rare_temp_patterns.temp_patterns[index_10].iloc[:,4:]
        d3 = d1 * d2 / 100
        d4 = pd.DataFrame({AEP_frequency_req[AEP_frequency_req.frequency == 'rare'].frequency_descriptor[index_11]:[d3]})
        list_temp.append(d4)
    list_temp = pd.concat(list_temp, axis=1)
    rare_hyetograph_depths.append(list_temp)

rare_hyetograph_depths = pd.concat(rare_hyetograph_depths, ignore_index=True)

# %%
rare_hyetograph_timestamp = []

for index_12 in rare_temp_patterns.index:
    list_temp = []
    e1 = rare_temp_patterns.temp_patterns[index_12].columns[4:]
    for index_13 in range(int(rare_temp_patterns.Duration[index_12] / timestep)):
        e2 = starting_date + index_13 * time_delta
        e3 = pd.DataFrame({e1[index_13]: [e2]})
        list_temp.append(e3)
    list_temp = pd.concat(list_temp, axis=1)
    e4 = pd.DataFrame({'increments': [list_temp]})
    rare_hyetograph_timestamp.append(e4)

rare_hyetograph_timestamp = pd.concat(rare_hyetograph_timestamp, ignore_index=True)

# %%
rare_rainseries = pd.concat([rare_temp_patterns.Duration_in_min, rare_temp_patterns.AEP, rare_hyetograph_timestamp, rare_hyetograph_depths], axis=1)

# %% [markdown]
# # list frequencies

# %% [markdown]
# ## frequent

# %%
freq_list = []

# for each column: frequency AEP column
for index_14 in freq_rainseries.columns[3:]:
    f0 = pd.DataFrame(freq_rainseries[index_14])

    # for each row: duration (9 categories), AEP (3 categories - 1 for this)
    for index_15 in f0.index:
        f1 = freq_rainseries.AEP[index_15]
        f2 = int(freq_rainseries.Duration_in_min[index_15])
        f3 = freq_rainseries.increments[index_15]
        f4 = f0[index_14][index_15]

        # for each row: temporal pattern (10 temporal patterns)
        for index_16 in f4.index:
            f5 = pd.DataFrame(f4.loc[index_16]).T
            temp_list = []

            # for each column: depending on duration - 2, 3, 4, 5, 6, 9, 12, 18, 24 timestamps
            for index_17 in f5.columns:
                f6 = dt.datetime.strftime(f3[index_17][0], '%m/%d/%Y %H:%M')
                f7 = f4[index_17][index_16]
                f8 = pd.DataFrame({'hydrograph':['{}      {}      {}'.format(rain_gauge, f6, f7)]})
                temp_list.append(f8)

            temp_list = pd.concat(temp_list, ignore_index=True)
            f9 = '{}__{}__{:03}__{:02}'.format(f1 if len(f1) == max_len_freq else f1 + (max_len_freq - len(f1)) * '_', 
                                               index_14 if len(index_14) == max_len_desc else index_14 + (max_len_desc - len(index_14)) * '_', 
                                               f2, index_16 + 1)
            f10 = pd.DataFrame({'frequency_AEP':[f1], 'AEP':[index_14], 'Duration':[f2], 'pattern':[index_16 + 1], 'name':[f9], 'timeserie': [temp_list]})
            freq_list.append(f10)

freq_list = pd.concat(freq_list, ignore_index=True)

# %% [markdown]
# ## intermediate

# %%
inte_list = []

# for each column: frequency AEP column
for index_14 in inte_rainseries.columns[3:]:
    f0 = pd.DataFrame(inte_rainseries[index_14])

    # for each row: duration (9 categories), AEP (3 categories - 1 for this)
    for index_15 in f0.index:
        f1 = inte_rainseries.AEP[index_15]
        f2 = int(inte_rainseries.Duration_in_min[index_15])
        f3 = inte_rainseries.increments[index_15]
        f4 = f0[index_14][index_15]

        # for each row: temporal pattern (10 temporal patterns)
        for index_16 in f4.index:
            f5 = pd.DataFrame(f4.loc[index_16]).T
            temp_list = []

            # for each column: depending on duration - 2, 3, 4, 5, 6, 9, 12, 18, 24 timestamps
            for index_17 in f5.columns:
                f6 = dt.datetime.strftime(f3[index_17][0], '%m/%d/%Y %H:%M')
                f7 = f4[index_17][index_16]
                f8 = pd.DataFrame({'hydrograph':['{}      {}      {}'.format(rain_gauge, f6, f7)]})
                temp_list.append(f8)

            temp_list = pd.concat(temp_list, ignore_index=True)
            f9 = '{}__{}__{:03}__{:02}'.format(f1 if len(f1) == max_len_freq else f1 + (max_len_freq - len(f1)) * '_', 
                                               index_14 if len(index_14) == max_len_desc else index_14 + (max_len_desc - len(index_14)) * '_', 
                                               f2, index_16 + 1)
            f10 = pd.DataFrame({'frequency_AEP':[f1], 'AEP':[index_14], 'Duration':[f2], 'pattern':[index_16 + 1], 'name':[f9], 'timeserie': [temp_list]})
            inte_list.append(f10)

inte_list = pd.concat(inte_list, ignore_index=True)

# %% [markdown]
# ## rare

# %%
rare_list = []

# for each column: frequency AEP column
for index_14 in rare_rainseries.columns[3:]:
    f0 = pd.DataFrame(rare_rainseries[index_14])

    # for each row: duration (9 categories), AEP (3 categories - 1 for this)
    for index_15 in f0.index:
        f1 = rare_rainseries.AEP[index_15]
        f2 = int(rare_rainseries.Duration_in_min[index_15])
        f3 = rare_rainseries.increments[index_15]
        f4 = f0[index_14][index_15]

        # for each row: temporal pattern (10 temporal patterns)
        for index_16 in f4.index:
            f5 = pd.DataFrame(f4.loc[index_16]).T
            temp_list = []

            # for each column: depending on duration - 2, 3, 4, 5, 6, 9, 12, 18, 24 timestamps
            for index_17 in f5.columns:
                f6 = dt.datetime.strftime(f3[index_17][0], '%m/%d/%Y %H:%M')
                f7 = f4[index_17][index_16]
                f8 = pd.DataFrame({'hydrograph':['{}      {}      {}'.format(rain_gauge, f6, f7)]})
                temp_list.append(f8)

            temp_list = pd.concat(temp_list, ignore_index=True)
            f9 = '{}__{}__{:03}__{:02}'.format(f1 if len(f1) == max_len_freq else f1 + (max_len_freq - len(f1)) * '_', 
                                               index_14 if len(index_14) == max_len_desc else index_14 + (max_len_desc - len(index_14)) * '_', 
                                               f2, index_16 + 1)
            f10 = pd.DataFrame({'frequency_AEP':[f1], 'AEP':[index_14], 'Duration':[f2], 'pattern':[index_16 + 1], 'name':[f9], 'timeserie': [temp_list]})
            rare_list.append(f10)

rare_list = pd.concat(rare_list, ignore_index=True)

# %% [markdown]
# # generate outputs

# %% [markdown]
# ## output directory

# %%
if not os.path.exists('../' + output_directory):
    os.makedirs('../' + output_directory)

# %% [markdown]
# ## frequent

# %%
for index_18 in freq_list.index:
    g1 = '../' + output_directory + '/' + freq_list.name[index_18] + '.inp'
    shutil.copyfile(orig_SWMM_file, g1)
    g2 = freq_list.timeserie[index_18]
    for index_19 in g2.index:
        g3 = g2.hydrograph[index_19]
        with open(g1, 'r+') as SWMM_file:
            g4 = SWMM_file.readlines()
            g4.insert(704 + index_19, g3 + '\n')
            SWMM_file.seek(0)
            SWMM_file.writelines(g4)

# %% [markdown]
# ## intermediate

# %%
for index_18 in inte_list.index:
    g1 = '../' + output_directory + '/' + inte_list.name[index_18] + '.inp'
    shutil.copyfile(orig_SWMM_file, g1)
    g2 = inte_list.timeserie[index_18]
    for index_19 in g2.index:
        g3 = g2.hydrograph[index_19]
        with open(g1, 'r+') as SWMM_file:
            g4 = SWMM_file.readlines()
            g4.insert(704 + index_19, g3 + '\n')
            SWMM_file.seek(0)
            SWMM_file.writelines(g4)

# %% [markdown]
# ## rare

# %%
for index_18 in rare_list.index:
    g1 = '../' + output_directory + '/' + rare_list.name[index_18] + '.inp'
    shutil.copyfile(orig_SWMM_file, g1)
    g2 = rare_list.timeserie[index_18]
    for index_19 in g2.index:
        g3 = g2.hydrograph[index_19]
        with open(g1, 'r+') as SWMM_file:
            g4 = SWMM_file.readlines()
            g4.insert(704 + index_19, g3 + '\n')
            SWMM_file.seek(0)
            SWMM_file.writelines(g4)
