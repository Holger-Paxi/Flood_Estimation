# %%
import pandas as pd
import numpy as np
import math
import shutil
import glob
import os
import gc
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import multiprocessing as mp
import pyswmm
from pyswmm import Simulation, Links
import swmmio
import swmmio.utils.modify_model

matplotlib.use(backend='agg')

# %%
def main():

    # %% [markdown]
    # # inputs

    # %%
    input_idf_table_file_path = './Assignment_3__Input_FE/Data_Centroid/BOM_IDF_Data/depths_-33.8774_151.093_all_design.csv'
    input_temp_pattern_file_path = './Assignment_3__Input_FE/Data_Centroid/ECsouth/ECsouth_Increments.csv'
    input_storm_stats_file_path = './Assignment_3__Input_FE/Data_Centroid/ECsouth/ECsouth_AllStats.csv'
    output_dir = './Assignment_3__Output_FE__1st__centroid/'
    original_inp_file_path = './Assignment_3__Input_FE/971007_SW5.INP'
    simulation_starting_time = '2000/01/01 00:00:00'
    stopping_time_after_precipitation_finish = {'days':0, 'hours':12, 'minutes':0, 'seconds':0}
    report_step = {'days':0, 'hours':0, 'minutes':5, 'seconds':0}
    wet_step = {'days':0, 'hours':0, 'minutes':5, 'seconds':0}
    dry_step = {'days':0, 'hours':0, 'minutes':5, 'seconds':0}
    routing_step = {'days':0, 'hours':0, 'minutes':0, 'seconds':30}
    link_to_get_results = 116
    timeseries_name = 'RainGauge'

    simulation_starting_time = pd.to_datetime(simulation_starting_time)
    stopping_time_after_precipitation_finish = pd.Timedelta(**stopping_time_after_precipitation_finish)
    report_step = pd.Timedelta(**report_step)
    wet_step = pd.Timedelta(**wet_step)
    dry_step = pd.Timedelta(**dry_step)
    routing_step = pd.Timedelta(**routing_step)
    link_to_get_results = str(link_to_get_results)

    print(
        input_idf_table_file_path,
        input_temp_pattern_file_path,
        input_storm_stats_file_path,
        original_inp_file_path,
        simulation_starting_time,
        stopping_time_after_precipitation_finish,
        report_step,
        wet_step,
        dry_step,
        routing_step,
        link_to_get_results,
        timeseries_name,
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

    # %%
    inp_file_dir = '{}inp_file_dir/'.format(output_dir)
    create_output_dir(inp_file_dir)
    inp_file_dir

    # %%
    rain_pattern_cum_dir = '{}rain_pattern_cum_dir/'.format(output_dir)
    create_output_dir(rain_pattern_cum_dir)
    rain_pattern_cum_dir

    # %%
    precipitation_dir = '{}precipitation_dir/'.format(output_dir)
    create_output_dir(precipitation_dir)
    precipitation_dir

    # %%
    flow_rate_dir = '{}flow_rate_dir/'.format(output_dir)
    create_output_dir(flow_rate_dir)
    flow_rate_dir

    # %%
    graphs_pattern_dir = '{}graphs_pattern_dir/'.format(output_dir)
    create_output_dir(graphs_pattern_dir)
    graphs_pattern_dir

    # %%
    graphs_frequency_dir = '{}graphs_frequency_dir/'.format(output_dir)
    create_output_dir(graphs_frequency_dir)
    graphs_frequency_dir

    # %%
    results_frequency_dir = '{}results_frequency_dir/'.format(output_dir)
    create_output_dir(results_frequency_dir)
    results_frequency_dir

    # %%
    results_stats_dir = '{}results_stats_dir/'.format(output_dir)
    create_output_dir(results_stats_dir)
    results_stats_dir

    # %% [markdown]
    # # Threads

    # %%
    n_core = mp.cpu_count() - 1
    n_core

    # %% [markdown]
    # # setting up the inp file

    # %%
    inp_file_path = '{}inp_template.inp'.format(output_dir)
    inp_file_path

    # %%
    # copy the original inp file
    shutil.copyfile(src=original_inp_file_path, dst=inp_file_path)

    # %%
    options_inp_file = swmmio.utils.dataframes.dataframe_from_inp(inp_path=inp_file_path, section='[OPTIONS]')
    options_inp_file.at['START_DATE', 'Value'] = simulation_starting_time.strftime(format='%m/%d/%Y')
    options_inp_file.at['START_TIME', 'Value'] = simulation_starting_time.strftime(format='%H:%M:%S')
    options_inp_file.at['REPORT_START_DATE', 'Value'] = simulation_starting_time.strftime(format='%m/%d/%Y')
    options_inp_file.at['REPORT_START_TIME', 'Value'] = simulation_starting_time.strftime(format='%H:%M:%S')
    options_inp_file.at['END_DATE', 'Value'] = (simulation_starting_time + stopping_time_after_precipitation_finish).strftime(format='%m/%d/%Y')
    options_inp_file.at['END_TIME', 'Value'] = (simulation_starting_time + stopping_time_after_precipitation_finish).strftime(format='%H:%M:%S')
    options_inp_file.at['REPORT_STEP', 'Value'] = str(report_step).split(sep=' ')[-1]
    options_inp_file.at['WET_STEP', 'Value'] = str(wet_step).split(sep=' ')[-1]
    options_inp_file.at['DRY_STEP', 'Value'] = str(dry_step).split(sep=' ')[-1]
    options_inp_file.at['ROUTING_STEP', 'Value'] = str(routing_step).split(sep=' ')[-1]
    options_inp_file.at['THREADS', 'Value'] = n_core

    options_inp_file

    # %%
    raingauges_inp_file = swmmio.utils.dataframes.dataframe_from_inp(inp_path=inp_file_path, section='[RAINGAGES]')
    raingauges_inp_file = raingauges_inp_file.iloc[:1]
    raingauges_inp_file.index = pd.Index(data=[timeseries_name], name='Name')
    raingauges_inp_file.at[timeseries_name, 'TimeIntrvl'] = str(pd.Timedelta(minutes=5)).split(sep=' ')[-1].rsplit(sep=':', maxsplit=1)[0]
    raingauges_inp_file.at[timeseries_name, 'DataSourceName'] = timeseries_name

    raingauges_inp_file

    # %%
    subcatchments_inp_file = swmmio.utils.dataframes.dataframe_from_inp(inp_path=inp_file_path, section='[SUBCATCHMENTS]')
    subcatchments_inp_file.Raingage = timeseries_name

    subcatchments_inp_file

    # %%
    timeseries_inp_file = swmmio.utils.dataframes.dataframe_from_inp(inp_path=inp_file_path, section='[TIMESERIES]')
    timeseries_inp_file.reset_index(inplace=True)
    timeseries_inp_file.drop(index=timeseries_inp_file.index[1:], inplace=True)
    timeseries_inp_file.at[0, 'Name'] = timeseries_name
    timeseries_inp_file.set_index(keys='Name', inplace=True)
    timeseries_inp_file.at[timeseries_name, 'Date'] = simulation_starting_time.strftime(format='%m/%d/%Y')
    timeseries_inp_file.at[timeseries_name, 'Time'] = simulation_starting_time.strftime(format='%H:%M')
    timeseries_inp_file.at[timeseries_name, 'Value'] = 0.0

    timeseries_inp_file

    # %%
    [swmmio.utils.modify_model.replace_inp_section(inp_path=inp_file_path, modified_section_header=ind1, new_data=ind2)
    for ind1, ind2 in zip(
        ['[OPTIONS]', '[RAINGAGES]', '[SUBCATCHMENTS]', '[TIMESERIES]'], 
        [options_inp_file, raingauges_inp_file, subcatchments_inp_file, timeseries_inp_file]
        )]

    # %% [markdown]
    # # building idf table

    # %%
    idf_table = pd.read_csv(filepath_or_buffer=input_idf_table_file_path, skiprows=9)
    frequency_label = idf_table.columns[2:].to_numpy()
    idf_table.columns = idf_table.columns.to_series().apply(
        func=lambda arg: 
            '_'.join(arg.split(sep=' ')) 
            if arg[:8] == 'Duration' 
            else 'freq_' + '_'.join('_perc'.join('_'.join(arg.split(sep=' ')).split(sep='%')).split(sep='.'))
        )
    frequency_tag = idf_table.columns[2:].to_numpy()
    idf_table[['Duration', 'units']] = idf_table.apply(func=lambda arg: arg.Duration.split(sep=' '), axis=1, result_type='expand')
    idf_table = idf_table[idf_table.columns[[0,-1] + list(range(1,len(idf_table.columns) - 1))]]
    idf_table.Duration = idf_table.Duration.astype(float)
    idf_table.to_csv(path_or_buf='{}table_idf_depths.csv'.format(output_dir), index=False)
    idf_table.to_parquet(path='{}table_idf_depths.parquet'.format(output_dir))

    idf_table

    # %% [markdown]
    # # building temporal pattern table

    # %%
    temp_pattern_table = pd.read_csv(filepath_or_buffer=input_temp_pattern_file_path)
    temp_pattern_table.rename(columns=lambda arg: arg.strip(), inplace=True)
    # temp_pattern_table.rename(columns={'Duration':'Duration_in_min'}, inplace=True)
    temp_pattern_table.columns = pd.Index(data=temp_pattern_table.columns[:5].to_list() + ['Increment_{:02}'.format(ind) for ind in range(len(temp_pattern_table.columns[5:]))])
    temp_pattern_table.to_csv(path_or_buf='{}table_temp_patterns.csv'.format(output_dir), index=False)
    temp_pattern_table.to_parquet(path='{}table_temp_patterns.parquet'.format(output_dir))

    temp_pattern_table

    # %%
    storm_stats_table = pd.read_csv(filepath_or_buffer=input_storm_stats_file_path)
    storm_stats_table.rename(columns=lambda arg: arg.strip(), inplace=True)
    for ind in [(' ','_'), ('(',''), (')',''), ('%','in_perc'), ('_No.','_Number'), ('_min','_in_min'), ('_mm','_in_mm')]:
        storm_stats_table.rename(columns=lambda arg: arg.replace(*ind), inplace=True)
    # storm_stats_table.rename(columns={'Burst_Duration_in_min':'Duration_in_min'}, inplace=True)
    storm_stats_table.dropna(axis=0, inplace=True)
    storm_stats_table.Burst_Start_Date = pd.to_datetime(arg=storm_stats_table.Burst_Start_Date, format='%m/%d/%Y %H:%M')
    storm_stats_table.Burst_End_Date = pd.to_datetime(arg=storm_stats_table.Burst_End_Date, format='%m/%d/%Y %H:%M')
    storm_stats_table.Event_ID = storm_stats_table.Event_ID.astype(int)
    storm_stats_table.Burst_Duration_in_min = storm_stats_table.Burst_Duration_in_min.astype(int)
    storm_stats_table.Burst_Loading = storm_stats_table.Burst_Loading.astype(int)
    storm_stats_table.DB_Event_Reference_Number = storm_stats_table.DB_Event_Reference_Number.astype(int)
    storm_stats_table.DB_Pluviograph_Reference_Number = storm_stats_table.DB_Pluviograph_Reference_Number.astype(int)
    storm_stats_table.to_csv(path_or_buf='{}table_storm_stats.csv'.format(output_dir), index=False)
    storm_stats_table.to_parquet(path='{}table_storm_stats.parquet'.format(output_dir))

    storm_stats_table

    # %% [markdown]
    # # building frequency classification table

    # %%
    ey_constants = np.array(object=[12,6,4,3,2,1,0.5,0.2])
    ey_constants

    # %%
    aep_variables = np.multiply(np.add(1, np.negative(np.exp(np.negative(ey_constants)))),100)
    aep_variables

    # %%
    aep_constants = np.array(object=[50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02])
    aep_constants

    # %%
    ey_variables = np.negative(np.log(np.add(1, np.negative(np.divide(aep_constants, 100)))))
    ey_variables

    # %%
    ey = np.flip(m=np.sort(a=np.concatenate((ey_constants, ey_variables))))
    ey

    # %%
    aep_percentage = np.flip(m=np.sort(a=np.concatenate((aep_constants, aep_variables))))
    aep_percentage

    # %%
    aep_1_in_x = np.divide(100, aep_percentage)
    aep_1_in_x

    # %%
    ari = np.divide(1,ey)
    ari

    # %%
    frequency_window = np.array(object=['frequent', 'intermediate', 'rare'])[
        np.add(np.digitize(x=aep_percentage, bins=np.array(object=[100, 14.4, 3.2, 0]), right=False), np.negative(1))
        ]
    frequency_window

    # %%
    frequency_tag = np.concatenate((frequency_tag, np.array(object=['freq_1_in_5000'])))
    frequency_tag

    # %%
    frequency_label = np.concatenate((frequency_label, np.array(object=['1 in 5000'])))
    frequency_label

    # %%
    frequency_table = pd.DataFrame(
        data=np.column_stack(tup=(ey, aep_percentage, aep_1_in_x, ari, frequency_window, frequency_label, frequency_tag)),
        columns=['EY', 'AEP_percentage', 'AEP_1_in_x', 'ARI', 'freq_window', 'freq_label', 'freq_tag']
        )
    frequency_table.to_csv(path_or_buf='{}table_frequency.csv'.format(output_dir), index=False)
    frequency_table.to_parquet(path='{}table_frequency.parquet'.format(output_dir))
    frequency_table

    # %%
    r_ey = ey.copy()
    r_ey[:14] = r_ey[:14].round(decimals=2)
    r_ey[14:17] = r_ey[14:17].round(decimals=3)
    r_ey[17:] = r_ey[17:].round(decimals=4)
    r_ey

    # %%
    r_aep_percentage = aep_percentage.copy()
    r_aep_percentage = r_aep_percentage.round(decimals=2)
    r_aep_percentage

    # %%
    r_aep_1_in_x = aep_1_in_x.copy()
    r_aep_1_in_x[:2] = r_aep_1_in_x[:2].round(decimals=3)
    r_aep_1_in_x[2:] = r_aep_1_in_x[2:].round(decimals=2)
    r_aep_1_in_x

    # %%
    r_ari = ari.copy()
    r_ari = r_ari.round(decimals=2)
    r_ari

    # %%
    r_frequency_table = pd.DataFrame(
        data=np.column_stack(tup=(r_ey, r_aep_percentage, r_aep_1_in_x, r_ari, frequency_window, frequency_label, frequency_tag)),
        columns=['EY', 'AEP_percentage', 'AEP_1_in_x', 'ARI', 'freq_window', 'freq_label', 'freq_tag']
        )
    r_frequency_table.to_csv(path_or_buf='{}table_rounded_frequency.csv'.format(output_dir), index=False)
    r_frequency_table.to_parquet(path='{}table_rounded_frequency.parquet'.format(output_dir))
    r_frequency_table

    # %% [markdown]
    # # merge stats and pattern tables into timeseries

    # %%
    df_storm_stats = storm_stats_table.copy()
    df_storm_stats.drop(columns=storm_stats_table.columns[[1,2,4,5,7,8,9,10,11,12,13,14]], inplace=True)
    df_storm_stats.rename(columns={'Event_ID':'event_id', 'Burst_Duration_in_min':'duration_in_min', 'AEP_Window':'freq_window'}, inplace=True)
    df_storm_stats['id_duration'] = pd.Series(data=np.unique(ar=df_storm_stats.duration_in_min.to_numpy(), return_inverse=True)[1])
    df_storm_stats['id_window'] = pd.Series(data=np.unique(ar=df_storm_stats.freq_window.to_numpy(), return_inverse=True)[1])
    df_storm_stats['id_pattern'] = pd.Series(data=list(np.arange(stop=10))*int(len(df_storm_stats)/10))

    df_storm_stats

    # %%
    df_temp_pattern = temp_pattern_table.copy()
    df_temp_pattern['time_series'] = df_temp_pattern.apply(func=lambda arg: [ind for ind in arg.to_list()[5:] if str(ind) != 'nan'], axis=1)
    df_temp_pattern.drop(columns=df_temp_pattern.columns[5:-1], inplace=True)
    df_temp_pattern.drop(columns='Region', inplace=True)
    df_temp_pattern.rename(columns={'EventID':'event_id', 'Duration':'duration_in_min', 'TimeStep':'time_step', 'AEP':'freq_window'}, inplace=True)
    df_temp_pattern['id_duration'] = pd.Series(data=np.unique(ar=df_temp_pattern.duration_in_min.to_numpy(), return_inverse=True)[1])
    df_temp_pattern['id_timestep'] = pd.Series(data=np.unique(ar=df_temp_pattern.time_step.to_numpy(), return_inverse=True)[1])
    df_temp_pattern['id_window'] = pd.Series(data=np.unique(ar=df_temp_pattern.freq_window.to_numpy(), return_inverse=True)[1])
    df_temp_pattern['id_pattern'] = pd.Series(data=list(np.arange(stop=10))*int(len(df_temp_pattern)/10))

    df_temp_pattern

    # %%
    df_timeseries = pd.merge(left=df_storm_stats, right=df_temp_pattern, how='inner', on='event_id', suffixes=(None, '__del'))# left_on='Event_ID', right_on='EventID')
    df_timeseries.drop(columns=[ind for ind in df_timeseries.columns.to_list() if ind.rsplit(sep='__', maxsplit=1)[-1] == 'del'], inplace=True)

    df_timeseries

    # %% [markdown]
    # # idf depths

    # %%
    df_map = df_timeseries.copy()
    df_map = df_map[['duration_in_min', 'time_step']]
    df_map.reset_index(drop=True, inplace=True)
    df_map = {key:val for key,val in zip(df_map.duration_in_min, df_map.time_step)}

    df_map

    # %%
    df_id = frequency_table.copy()
    df_id.drop(index=len(df_id)-1, inplace=True)
    df_id = df_id[['freq_window', 'freq_label', 'freq_tag']]#.iloc[:-1,4:]
    df_id = pd.concat(objs=[
        pd.Series(data=np.full(shape=len(df_id), fill_value='depth')),
        df_id,
        pd.Series(data=np.unique(ar=df_id.freq_window.to_numpy(), return_inverse=True)[1], name='id_window'),
        df_id.index.to_series(name='id_tag')
        ], axis=1)

    df_id

    # %%
    df_idf = idf_table.copy()
    df_idf = df_idf[df_idf.Duration_in_min >= 10]
    df_idf.reset_index(drop=True, inplace=True)
    df_idf.rename(columns={'Duration':'duration', 'Duration_in_min':'duration_in_min'}, inplace=True)
    df_idf.duration_in_min = df_idf.duration_in_min.astype(int)
    df_idf['id_duration'] = pd.Series(data=np.unique(ar=df_idf.duration_in_min.to_numpy(), return_inverse=True)[1])
    df_idf['time_step'] = df_idf.duration_in_min.map(arg=df_map)
    df_idf['id_timestep'] = pd.Series(data=np.unique(ar=df_idf.time_step.to_numpy(), return_inverse=True)[1])
    df_idf.set_index(keys=[ind for ind in df_idf.columns.to_list() if ind.split(sep='_', maxsplit=1)[0] == 'id'] + ['duration', 'units', 'duration_in_min', 'time_step'] , inplace=True)
    df_idf.columns = pd.MultiIndex.from_frame(df=df_id)
    df_idf = df_idf.stack(level=list(df_idf.columns.names)[1:], future_stack=True)
    df_idf.reset_index(inplace=True)

    df_idf

    # %% [markdown]
    # # rainfall_patterns

    # %%
    rain_data = pd.merge(left=df_timeseries, right=df_idf, how='cross', suffixes=[None, '__del'])
    rain_data = rain_data[
        (rain_data.id_window == rain_data.id_window__del) & 
        (rain_data.id_duration == rain_data.id_duration__del) & 
        (rain_data.id_timestep == rain_data.id_timestep__del)
        ]
    rain_data.drop(columns=[ind for ind in rain_data.columns.to_list() if ind.rsplit(sep='__', maxsplit=1)[-1] == 'del'], inplace=True)
    rain_data.drop(columns='event_id', inplace=True)
    rain_data.drop_duplicates(subset=list(rain_data.columns.to_numpy()[~np.isin(element=rain_data.columns.to_numpy(), test_elements='time_series')]), inplace=True)
    rain_data.sort_values(by=['id_tag', 'id_duration', 'id_timestep', 'id_pattern'], inplace=True)
    rain_data.reset_index(drop=True, inplace=True)
    rain_data['id_group'] = pd.Series(data=np.concatenate([np.full(shape=10, fill_value=ind) for ind in np.arange(stop=np.divide(rain_data.shape[0], 10), dtype=np.int64)]))
    rain_data['rain_label'] = (
        (rain_data.id_group.astype(str).str.len().max() - rain_data.id_group.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_group)) + rain_data.id_group.astype(str) + '__' +
        (rain_data.index.astype(str).str.len().max() - rain_data.index.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.index)) + rain_data.index.astype(str) + '__' +
        (rain_data.id_tag.astype(str).str.len().max() - rain_data.id_tag.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_tag)) + rain_data.id_tag.astype(str) + '_' +
        (rain_data.id_duration.astype(str).str.len().max() - rain_data.id_duration.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_duration)) + rain_data.id_duration.astype(str) + '_' +
        (rain_data.id_timestep.astype(str).str.len().max() - rain_data.id_timestep.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_timestep)) + rain_data.id_timestep.astype(str) + '_' +
        (rain_data.id_pattern.astype(str).str.len().max() - rain_data.id_pattern.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_pattern)) + rain_data.id_pattern.astype(str) + '__' +
        (rain_data.id_window.astype(str).str.len().max() - rain_data.id_window.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_window)) + rain_data.id_window.astype(str) + '__' +
        (rain_data.freq_window.str.len().max() - rain_data.freq_window.str.len())*pd.Series(data=['_']*len(rain_data.freq_window)) + rain_data.freq_window + '__' +
        (rain_data.freq_tag.str.len().max() - rain_data.freq_tag.str.len())*pd.Series(data=['_']*len(rain_data.freq_tag)) + rain_data.freq_tag + '__' +
        (rain_data.duration_in_min.astype(str).str.len().max() - rain_data.duration_in_min.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.duration_in_min)) + rain_data.duration_in_min.astype(str) + '__' +
        (rain_data.time_step.astype(str).str.len().max() - rain_data.time_step.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.time_step)) + rain_data.time_step.astype(str) + '__' +
        (rain_data.id_pattern.astype(str).str.len().max() - rain_data.id_pattern.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_pattern)) + rain_data.id_pattern.astype(str)
        )
    rain_data.time_series = rain_data.time_series.apply(func=lambda arg: [0] + arg + [0])

    rain_data

    # %% [markdown]
    # # inp files

    # %%
    def create_inp_files(
        arg_rain_label, arg_time_series, arg_depth, arg_time_step,
        arg_sim_starting_time, arg_timeseries_name,
        arg_inp_file_dir, arg_inp_file_path, arg_additional_rain_stopping_time
        ):

        # create inp file
        inp_file = '{}inp_file__{}.inp'.format(arg_inp_file_dir, arg_rain_label)
        shutil.copyfile(src=arg_inp_file_path, dst=inp_file)

        # [TIMESERIES] section
        rain_value = pd.Series(
            data=np.divide(np.multiply(arg_depth, np.array(object=arg_time_series)), 100), name='Value')

        rain_date_time = pd.Series(data=pd.date_range(
            start=arg_sim_starting_time,
            periods=rain_value.size,
            freq=pd.Timedelta(minutes=arg_time_step)
            ), name='date_time')
        rain_date = rain_date_time.dt.strftime(date_format='%m/%d/%Y')
        rain_date.name = 'Date'
        rain_time = rain_date_time.dt.strftime(date_format='%H:%M')
        rain_time.name = 'Time'

        rain_timeseries_name = pd.Series(data=[arg_timeseries_name]*rain_value.size, name='Name')

        timeseries_section = pd.concat(objs=[rain_timeseries_name, rain_date, rain_time, rain_value], axis=1)
        timeseries_section.set_index(keys='Name', inplace=True)

        # [OPTIONS] section
        end_date_time = rain_date_time.iloc[-1] + arg_additional_rain_stopping_time
        end_date = end_date_time.strftime(format='%m/%d/%Y')
        end_time = end_date_time.strftime(format='%H:%M:%S')

        options_section = swmmio.utils.dataframes.dataframe_from_inp(inp_path=inp_file, section='[OPTIONS]')
        options_section.at['END_DATE', 'Value'] = end_date
        options_section.at['END_TIME', 'Value'] = end_time

        # [RAINGAGES] section
        time_interval = str(pd.Timedelta(minutes=arg_time_step)).split(sep=' ')[-1].rsplit(sep=':', maxsplit=1)[0]

        raingauges_section = swmmio.utils.dataframes.dataframe_from_inp(inp_path=inp_file, section='[RAINGAGES]')
        raingauges_section.at[arg_timeseries_name, 'TimeIntrvl'] = time_interval

        # replace section into SWMM files
        [swmmio.utils.modify_model.replace_inp_section(inp_path=inp_file, modified_section_header=ind1, new_data=ind2)
        for ind1, ind2 in zip(
            ['[OPTIONS]', '[RAINGAGES]', '[TIMESERIES]'], 
            [options_section, raingauges_section, timeseries_section]
            )]

    # %%
    rain_data.apply(func=lambda arg: create_inp_files(
        arg.rain_label, arg.time_series, arg.depth, arg.time_step, 
        simulation_starting_time, timeseries_name,
        inp_file_dir, inp_file_path, stopping_time_after_precipitation_finish
        ), axis=1)

    # %% [markdown]
    # # run SWMM

    # %%
    def run_swmm(arg_inp_dir, arg_link, arg_label):

        model = '{}inp_file__{}.inp'.format(arg_inp_dir, arg_label)

        with Simulation(inputfile=model) as sim:
            link_sim = Links(model=sim)[arg_link]
            time_stamp = []
            flow_rate = []
            for ind2, step in enumerate(sim):
                if (sim.current_time.minute%5 == 0)&(sim.current_time.second == 0):
                    time_stamp.append(sim.current_time)
                    flow_rate.append(link_sim.flow)
            
            df_flow_data = pd.concat(
                objs=[pd.Series(data=time_stamp, name='timestamp'), pd.Series(data=flow_rate, name='flow_rate')], 
                axis=1)
            df_flow_data.set_index(keys='timestamp', inplace=True)

            return [max(flow_rate), df_flow_data]

    # %%
    def run_multiprocessing(arg_tuples_from_label_series, arg_n_core=n_core, arg_max_tasks_per_child=100):

        n_task = len(arg_tuples_from_label_series)
        n_task_per_chunk = math.ceil(n_task/arg_n_core)

        with mp.Pool(processes=arg_n_core, maxtasksperchild=arg_max_tasks_per_child) as pool:
            outflow = [ind for ind in pool.starmap(func=run_swmm, iterable=arg_tuples_from_label_series, chunksize=n_task_per_chunk)]

        return outflow

    # %%
    rain_data[['max_flow_rate', 'flow_data']] = pd.DataFrame(data=run_multiprocessing(
        [(inp_file_dir, link_to_get_results, ind) for ind in rain_data.rain_label.to_list()], n_core))
    rain_data

    # %% [markdown]
    # # merging and rearrenging dataframes

    # %%
    def temp_pattern(arg_time_series, arg_time_step, arg_sim_start_time):

        cum_pattern = pd.Series(
            data=np.array(object=arg_time_series)[:-1].cumsum(), 
            name='cum_pattern')
        pat_timestamp = pd.Series(
            data=pd.date_range(
                start=arg_sim_start_time, 
                periods=cum_pattern.size, 
                freq=pd.Timedelta(minutes=arg_time_step)
                ), 
            name='timestamp')
        
        df_cum_pattern = pd.concat(objs=[pat_timestamp, cum_pattern], axis=1)
        df_cum_pattern.set_index(keys='timestamp', inplace=True)

        return df_cum_pattern

    # %%
    rain_data['cum_temp_pattern_data'] = rain_data.apply(func=lambda arg: temp_pattern(arg.time_series, arg.time_step, simulation_starting_time), axis=1)
    rain_data

    # %%
    def rainfall_depth(arg_time_series, arg_depth, arg_time_step, arg_sim_start_time):

        prec_depth = pd.Series(
            data=np.divide(np.multiply(arg_depth, np.array(object=arg_time_series)), 100),
            name='prec_depth')
        prec_timestamp = pd.Series(
            data=pd.date_range(
                start=arg_sim_start_time, 
                periods=prec_depth.size, 
                freq=pd.Timedelta(minutes=arg_time_step)
                ), 
            name='timestamp')

        df_precipitation = pd.concat(objs=[prec_timestamp, prec_depth], axis=1)
        df_precipitation.set_index(keys='timestamp', inplace=True)

        return df_precipitation

    # %%
    rain_data['rainfall_data'] = rain_data.apply(func=lambda arg: rainfall_depth(arg.time_series, arg.depth, arg.time_step, simulation_starting_time), axis=1)
    rain_data

    # %%
    rain_data.drop(columns='time_series', inplace=True)
    rain_data

    # %%
    rain_data = pd.DataFrame(data=rain_data.groupby(by=['id_tag', 'id_duration']), columns=['id_tag_dur', 'rainfall_data'])
    rain_data[['id_tag', 'id_duration']] = rain_data.apply(func=lambda arg: (arg.id_tag_dur[0], arg.id_tag_dur[1]), axis=1, result_type='expand')
    rain_data.drop(columns='id_tag_dur', inplace=True)

    rain_data

    # %%
    def rearrange_timeseries(arg_df_rainfall_data):

        arg_df_rainfall_data.drop(columns=['id_tag', 'id_duration'], inplace=True)

        ind_sep = np.isin(
            element=arg_df_rainfall_data.columns.to_numpy(), 
            test_elements=np.array(object=[
                'id_pattern', 'rain_label', 'max_flow_rate', 
                'flow_data', 'cum_temp_pattern_data', 'rainfall_data']))
        arg_df_rainfall_data, time_series_data = (
            arg_df_rainfall_data.copy()[list(arg_df_rainfall_data.columns.to_numpy()[~ind_sep])], 
            arg_df_rainfall_data.copy()[list(arg_df_rainfall_data.columns.to_numpy()[ind_sep])])
        arg_df_rainfall_data.drop_duplicates(inplace=True)
        arg_df_rainfall_data.reset_index(drop=True, inplace=True)
        time_series_data.reset_index(drop=True, inplace=True)

        max_flow_rate_data = time_series_data.max_flow_rate.to_numpy()

        flow_rate_data = pd.concat(objs=time_series_data.flow_data.to_list(), axis=1)
        flow_rate_data.columns = ['flow rate {:02}'.format(ind) for ind in np.arange(stop=flow_rate_data.columns.size)]
        flow_rate_data = flow_rate_data.iloc[:np.where(flow_rate_data.index.to_numpy() == flow_rate_data[flow_rate_data.sum(axis=1) != 0].iloc[-1].name)[0][0]+2, :]

        pattern_data = pd.concat(objs=time_series_data.cum_temp_pattern_data.to_list(), axis=1)
        pattern_data.columns = ['pattern {:02}'.format(ind) for ind in np.arange(stop=pattern_data.columns.size)]

        prec_data = pd.concat(objs=time_series_data.rainfall_data.to_list(), axis=1)
        prec_data.columns = ['rain depth {:02}'.format(ind) for ind in np.arange(stop=prec_data.columns.size)]

        arg_df_rainfall_data['max_flow_rate'] = pd.Series()
        arg_df_rainfall_data['flow_data'] = pd.Series()
        arg_df_rainfall_data['cum_pattern_data'] = pd.Series()
        arg_df_rainfall_data['prec_data'] = pd.Series()

        arg_df_rainfall_data.at[0, 'max_flow_rate'] = max_flow_rate_data
        arg_df_rainfall_data.at[0, 'flow_data'] = flow_rate_data
        arg_df_rainfall_data.at[0, 'cum_pattern_data'] = pattern_data
        arg_df_rainfall_data.at[0, 'prec_data'] = prec_data

        arg_df_rainfall_data = tuple(arg_df_rainfall_data.itertuples(index=False, name=None))[0]

        return arg_df_rainfall_data

    # %%
    rain_data[['duration_in_min', 'freq_window', 'id_window', 'time_step',
        'id_timestep', 'duration', 'units', 'freq_label', 'freq_tag', 
        'depth', 'id_group', 'max_flow_rate', 'flow_data', 
        'cum_pattern_data', 'prec_data']] = rain_data.apply(func=lambda arg: rearrange_timeseries(arg.rainfall_data), axis=1, result_type='expand')
    rain_data

    # %%
    rain_data.drop(columns='rainfall_data', inplace=True)
    rain_data

    # %%
    rain_data['label'] = (
        (rain_data.id_group.astype(str).str.len().max() - rain_data.id_group.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_group)) + rain_data.id_group.astype(str) + '__' +
        (rain_data.id_tag.astype(str).str.len().max() - rain_data.id_tag.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_tag)) + rain_data.id_tag.astype(str) + '_' +
        (rain_data.id_duration.astype(str).str.len().max() - rain_data.id_duration.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_duration)) + rain_data.id_duration.astype(str) + '_' +
        (rain_data.id_timestep.astype(str).str.len().max() - rain_data.id_timestep.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_timestep)) + rain_data.id_timestep.astype(str) + '__' +
        (rain_data.id_window.astype(str).str.len().max() - rain_data.id_window.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_window)) + rain_data.id_window.astype(str) + '__' +
        (rain_data.freq_window.str.len().max() - rain_data.freq_window.str.len())*pd.Series(data=['_']*len(rain_data.freq_window)) + rain_data.freq_window + '__' +
        (rain_data.freq_tag.str.len().max() - rain_data.freq_tag.str.len())*pd.Series(data=['_']*len(rain_data.freq_tag)) + rain_data.freq_tag + '__' +
        (rain_data.duration_in_min.astype(str).str.len().max() - rain_data.duration_in_min.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.duration_in_min)) + rain_data.duration_in_min.astype(str) + '__' +
        (rain_data.time_step.astype(str).str.len().max() - rain_data.time_step.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.time_step)) + rain_data.time_step.astype(str)
        )
    rain_data

    # %% [markdown]
    # # run multiprocessing for export data and plots

    # %%
    def export_data_and_plot(
        arg_freq_window, arg_freq_label, arg_duration_in_min, arg_time_step, 
        arg_id_group, arg_duration, arg_units, arg_depth, arg_label, 
        arg_cum_pattern_data, arg_prec_data, arg_flow_data,
        arg_rain_pattern_cum_dir, arg_precipitation_dir, arg_flow_rate_dir, arg_graphs_pattern_dir
        ):

        matplotlib.use(backend='agg')

        # export data
        arg_cum_pattern_data.to_parquet(path='{}cum_pattern__{}.parquet'.format(arg_rain_pattern_cum_dir, arg_label))
        arg_prec_data.to_parquet(path='{}prec_data__{}.parquet'.format(arg_precipitation_dir, arg_label))
        arg_flow_data.to_parquet(path='{}flow_data__{}.parquet'.format(arg_flow_rate_dir, arg_label))

        # plot
        fig, ax = plt.subplots(nrows=11, ncols=3, figsize=(10,40), num=1, clear=True)

        colors = list(mcolors.TABLEAU_COLORS.keys())

        # temporal patterns
        for col, color in zip(arg_cum_pattern_data.columns, colors):
            ax[0,0].plot(
                arg_cum_pattern_data.index.to_numpy(),
                arg_cum_pattern_data[col].to_numpy(),
                linewidth=2,
                color=color,
                label=col
                )

        ax[0,0].legend(fontsize=8, framealpha=0.5)
        ax[0,0].grid(visible=True, which='both')
        ax[0,0].set_title(label='Temporal Patterns', fontdict={'fontsize':10})
        ax[0,0].set_xlabel(xlabel='Timestamp', fontdict={'fontsize':9})
        ax[0,0].set_ylabel(ylabel='Percentage Precipitation', fontdict={'fontsize':9})
        ax[0,0].set_yticklabels(labels=ax[0,0].get_yticklabels(), fontdict={'fontsize':8})
        ax[0,0].set_xticklabels(labels=ax[0,0].get_xticklabels(), fontdict={'fontsize':8, 'rotation':'vertical'})

        for ind, col, color in zip(range(1,11), arg_cum_pattern_data.columns, colors):
            ax[ind,0].plot(
                arg_cum_pattern_data.index.to_numpy(),
                arg_cum_pattern_data[col].to_numpy(),
                linewidth=2,
                color=color,
                label=col
                )

            ax[ind,0].legend(fontsize=8, framealpha=0.5)
            ax[ind,0].grid(visible=True, which='both')
            ax[ind,0].set_title(label='Temporal Pattern: {}'.format(col.title()), fontdict={'fontsize':10})
            ax[ind,0].set_xlabel(xlabel='Timestamp', fontdict={'fontsize':9})
            ax[ind,0].set_ylabel(ylabel='Percentage Precipitation', fontdict={'fontsize':9})
            ax[ind,0].set_yticklabels(labels=ax[ind,0].get_yticklabels(), fontdict={'fontsize':8})
            ax[ind,0].set_xticklabels(labels=ax[ind,0].get_xticklabels(), fontdict={'fontsize':8, 'rotation':'vertical'})

        # precipitation data
        for col, color in zip(arg_prec_data.columns, colors):
            ax[0,1].fill_between(
                x=arg_prec_data.index.to_numpy(),
                y1=arg_prec_data[col].to_numpy(),
                step='post',
                linewidth=0.5,
                color=color,
                label=col
                )

        ax[0,1].legend(fontsize=8, framealpha=0.5)
        ax[0,1].grid(visible=True, which='both')
        ax[0,1].set_title(label='Precipitation Series', fontdict={'fontsize':10})
        ax[0,1].set_xlabel(xlabel='Timestamp', fontdict={'fontsize':9})
        ax[0,1].set_ylabel(ylabel='Precipitation Depth ($mm$)', fontdict={'fontsize':9})
        ax[0,1].set_yticklabels(labels=ax[0,1].get_yticklabels(), fontdict={'fontsize':8})
        ax[0,1].set_xticklabels(labels=ax[0,1].get_xticklabels(), fontdict={'fontsize':8, 'rotation':'vertical'})

        for ind, col, color in zip(range(1,11), arg_prec_data.columns, colors):
            ax[ind,1].fill_between(
                x=arg_prec_data.index.to_numpy(),
                y1=arg_prec_data[col].to_numpy(),
                step='post',
                linewidth=0.5,
                color=color,
                label=col
                )

            ax[ind,1].legend(fontsize=8, framealpha=0.5)
            ax[ind,1].grid(visible=True, which='both')
            ax[ind,1].set_title(label='Precipitation Serie: {}'.format(col.title()), fontdict={'fontsize':10})
            ax[ind,1].set_xlabel(xlabel='Timestamp', fontdict={'fontsize':9})
            ax[ind,1].set_ylabel(ylabel='Precipitation Depth ($mm$)', fontdict={'fontsize':9})
            ax[ind,1].set_yticklabels(labels=ax[ind,1].get_yticklabels(), fontdict={'fontsize':8})
            ax[ind,1].set_xticklabels(labels=ax[ind,1].get_xticklabels(), fontdict={'fontsize':8, 'rotation':'vertical'})

        # flow data
        for col, color in zip(arg_flow_data.columns, colors):
            ax[0,2].fill_between(
                x=arg_flow_data.index.to_numpy(),
                y1=arg_flow_data[col].to_numpy(),
                linewidth=0.5,
                color=color,
                label=col
                )

        ax[0,2].legend(fontsize=8, framealpha=0.5)
        ax[0,2].grid(visible=True, which='both')
        ax[0,2].set_title(label='Hydrographs', fontdict={'fontsize':10})
        ax[0,2].set_xlabel(xlabel='Timestamp', fontdict={'fontsize':9})
        ax[0,2].set_ylabel(ylabel='Flow Rate ($m^3/s$)', fontdict={'fontsize':9})
        ax[0,2].set_yticklabels(labels=ax[0,2].get_yticklabels(), fontdict={'fontsize':8})
        ax[0,2].set_xticklabels(labels=ax[0,2].get_xticklabels(), fontdict={'fontsize':8, 'rotation':'vertical'})

        for ind, col, color in zip(range(1,11), arg_flow_data.columns, colors):
            ax[ind,2].fill_between(
                x=arg_flow_data.index.to_numpy(),
                y1=arg_flow_data[col].to_numpy(),
                linewidth=0.5,
                color=color,
                label=col
                )

            ax[ind,2].legend(fontsize=8, framealpha=0.5)
            ax[ind,2].grid(visible=True, which='both')
            ax[ind,2].set_title(label='Hydrograph: {}'.format(col.title()), fontdict={'fontsize':10})
            ax[ind,2].set_xlabel(xlabel='Timestamp', fontdict={'fontsize':9})
            ax[ind,2].set_ylabel(ylabel='Flow Rate ($m^3/s$)', fontdict={'fontsize':9})
            ax[ind,2].set_yticklabels(labels=ax[0,2].get_yticklabels(), fontdict={'fontsize':8})
            ax[ind,2].set_xticklabels(labels=ax[0,2].get_xticklabels(), fontdict={'fontsize':8, 'rotation':'vertical'})

        fig.suptitle(
            t='ID Group: {}; Window Frequency: {}; Timestep: {}min\nFrequency: {}; Duration {}min ({} {}); Total Precipitation Depth:{}mm'.format(
                arg_id_group, arg_freq_window.title(), arg_time_step, arg_freq_label, arg_duration_in_min, arg_duration, arg_units, arg_depth), 
            x=0.5, y=1, fontsize=11)
        fig.tight_layout()
        fig.savefig(fname='{}freq_pattern__{}.png'.format(arg_graphs_pattern_dir, arg_label), bbox_inches='tight')

        plt.cla()
        plt.clf()
        plt.close(fig=fig) # suposedly, this line should be deleted

    # %%
    # USE THIS: when there is not enough memory instead of the next two blocks

    # rain_data.apply(func=lambda arg: export_data_and_plot(
    #     arg.freq_window, arg.freq_label, arg.duration_in_min, arg.time_step, 
    #     arg.id_group, arg.duration, arg.units, arg.depth, arg.label, 
    #     arg.cum_pattern_data, arg.prec_data, arg.flow_data,
    #     rain_pattern_cum_dir, precipitation_dir, flow_rate_dir, graphs_pattern_dir
    #     ), axis=1)

    # %%
    def run_multiprocessing_data_and_plots(arg_tuples_from_df, arg_n_core=n_core, arg_max_tasks_per_child=100):

        n_task = len(arg_tuples_from_df)
        n_task_per_chunk = math.ceil(n_task/arg_n_core)

        with mp.Pool(processes=arg_n_core, maxtasksperchild=arg_max_tasks_per_child) as pool:
            pool.starmap(func=export_data_and_plot, iterable=arg_tuples_from_df, chunksize=n_task_per_chunk)

    # %%
    run_multiprocessing_data_and_plots(
        [ind + [rain_pattern_cum_dir, precipitation_dir, flow_rate_dir, graphs_pattern_dir] for ind in rain_data[[
        'freq_window', 'freq_label', 'duration_in_min', 'time_step', 
        'id_group', 'duration', 'units', 'depth', 'label', 
        'cum_pattern_data', 'prec_data', 'flow_data']].values.tolist()],
        n_core
        )

    # %%
    rain_data.drop(columns=['flow_data', 'cum_pattern_data', 'prec_data'], inplace=True)
    rain_data

    # %%
    rain_data = pd.DataFrame(data=rain_data.groupby(by='id_tag'), columns=['id_tag', 'rain_data'])
    rain_data['id_window'] = rain_data.rain_data.apply(lambda arg: np.unique(ar=arg.id_window.to_numpy())[0])
    rain_data['freq_tag'] = rain_data.rain_data.apply(lambda arg: np.unique(ar=arg.freq_tag.to_numpy())[0])
    rain_data['freq_label'] = rain_data.rain_data.apply(lambda arg: np.unique(ar=arg.freq_label.to_numpy())[0])
    rain_data['freq_window'] = rain_data.rain_data.apply(lambda arg: np.unique(ar=arg.freq_window.to_numpy())[0])

    rain_data['id_duration'] = rain_data.rain_data.apply(lambda arg: arg.id_duration.to_numpy())
    rain_data['id_timestep'] = rain_data.rain_data.apply(lambda arg: arg.id_timestep.to_numpy())
    rain_data['id_group'] = rain_data.rain_data.apply(lambda arg: arg.id_group.to_numpy())

    rain_data['duration_in_min'] = rain_data.rain_data.apply(lambda arg: arg.duration_in_min.to_numpy())
    rain_data['duration'] = rain_data.rain_data.apply(lambda arg: arg.duration.to_numpy())
    rain_data['units'] = rain_data.rain_data.apply(lambda arg: arg.units.to_numpy())
    rain_data['time_step'] = rain_data.rain_data.apply(lambda arg: arg.time_step.to_numpy())

    rain_data['depth'] = rain_data.rain_data.apply(lambda arg: arg.depth.to_numpy())
    rain_data['max_flow_rate'] = rain_data.rain_data.apply(lambda arg: np.column_stack(tup=arg.max_flow_rate.to_list()))

    rain_data.drop(columns='rain_data', inplace=True)

    rain_data # 'freq_window', 'id_window', 'freq_label', 'freq_tag'

    # %%
    def col_index(arg_df):
        df_data_cols = pd.concat(
            objs=[
                pd.Series(data=arg_df[ind], name=ind) 
                for ind in [
                    'id_duration', 'id_timestep', 'id_group', 'duration_in_min', 
                    'duration', 'units', 'time_step', 'depth']], axis=1)
        df_data_cols = pd.MultiIndex.from_frame(df=df_data_cols)
        df_data_cols = pd.DataFrame(data=arg_df.max_flow_rate, columns=df_data_cols)

        return df_data_cols

    # %%
    rain_data.max_flow_rate = rain_data.apply(func=lambda arg: col_index(arg), axis=1)
    rain_data

    # %%
    rain_data.drop(
        columns=[
            'id_duration', 'id_timestep', 'id_group', 'duration_in_min', 
            'duration', 'units', 'time_step', 'depth'
            ], inplace=True)
    rain_data

    # %%
    rain_data['label'] = (
        (rain_data.id_tag.astype(str).str.len().max() - rain_data.id_tag.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_tag)) + rain_data.id_tag.astype(str) + '_' +
        (rain_data.id_window.astype(str).str.len().max() - rain_data.id_window.astype(str).str.len())*pd.Series(data=['0']*len(rain_data.id_window)) + rain_data.id_window.astype(str) + '__' +
        (rain_data.freq_window.str.len().max() - rain_data.freq_window.str.len())*pd.Series(data=['_']*len(rain_data.freq_window)) + rain_data.freq_window + '__' +
        (rain_data.freq_tag.str.len().max() - rain_data.freq_tag.str.len())*pd.Series(data=['_']*len(rain_data.freq_tag)) + rain_data.freq_tag
        )

    rain_data

    # %%
    def plot_and_export_freq_graphs(
        arg_max_flow_rate, arg_label, arg_id_tag, arg_freq_window, arg_freq_label,
        arg_results_frequency_dir, arg_results_stats_dir, arg_graphs_frequency_dir
        ):

        matplotlib.use(backend='agg')

        # export max flow data
        arg_max_flow_rate.to_parquet(path='{}max_flow_rate__{}.parquet'.format(arg_results_frequency_dir, arg_label))

        # decompress index to get data
        data_index = arg_max_flow_rate.columns.to_frame()
        data_index.reset_index(drop=True, inplace=True)

        # stats and export
        stats = pd.concat(objs=[arg_max_flow_rate.describe(), arg_max_flow_rate.quantile(q=[0.05,0.95])])
        stats.rename(index={0.05:'5%', 0.95:'95%'}, inplace=True)
        stats = stats.reindex(index=stats.index[[0,1,2,3,8,4,5,6,9,7]])
        stats.to_parquet(path='{}max_flow_rate_stats__{}.parquet'.format(arg_results_stats_dir, arg_label))

        # confidence limits
        cl_median_based = stats.iloc[:,np.where(stats.to_numpy() == stats.loc['50%'].max())[1][0]].to_dict()
        cl_mean_based = stats.iloc[:,np.where(stats.to_numpy() == stats.loc['mean'].max())[1][0]].to_dict()

        # table
        ax_table = np.vstack(
            tup=[
                data_index.duration_in_min.to_numpy(),
                data_index.time_step.to_numpy(),
                data_index.depth.to_numpy(),
                stats.to_numpy()
                ])

        # plot and export
        fig, ax = plt.subplots(figsize=(15,6), num=1, clear=True)

        ax.boxplot(
            x=arg_max_flow_rate.to_numpy(),
            positions=data_index.id_duration.to_numpy(),
            showmeans=True,
            flierprops={'marker':'x', 'markeredgecolor':'steelblue'},
            medianprops={'linestyle':'-', 'color':'steelblue'},
            meanprops={'marker':'x', 'markeredgecolor':'red'}
            );

        ax.grid(visible=True, which='both')
        ax.set_xticks(ticks=data_index.id_duration.to_numpy())
        ax.set_xticklabels(
            labels=[
                '{} {}'.format(ind1, ind2) for ind1, ind2 in 
                zip(data_index.duration.to_numpy(), data_index.units.to_numpy())], 
            rotation='vertical', fontdict={'fontsize':8, 'rotation':'vertical'});
        ax.set_yticklabels(labels=ax.get_yticklabels(), fontdict={'fontsize':8});

        table = ax.table(
            cellText=ax_table.round(decimals=3),
            rowLabels=np.hstack((np.array(object=['duration', 'timestep', 'prec depth']), stats.index.to_numpy())),
            bbox=[0,-0.72,1,0.5])
        table.set_fontsize(8)

        ax.set_title(
            label='ID: {}; Window Frequency: {}; Frequency: {}'.format(
                arg_id_tag, arg_freq_window.title(), arg_freq_label
                ), 
            fontdict={'fontsize':10})
        ax.set_xlabel(xlabel='Duration ($min$)', fontdict={'fontsize':9})
        ax.set_ylabel(ylabel='Flow Rate ($m^3/s$)', fontdict={'fontsize':9})

        fig.savefig(
            fname='{}freq_graph__{}.png'.format(arg_graphs_frequency_dir, arg_label),
            bbox_inches='tight')
        
        plt.cla()
        plt.clf()
        plt.close(fig=fig)

        return cl_mean_based, cl_median_based

    # %%
    rain_data[['cl_mean_based', 'cl_median_based']] = rain_data.apply(
        func=lambda arg: plot_and_export_freq_graphs(
        arg.max_flow_rate, arg.label, arg.id_tag, arg.freq_window, arg.freq_label,
        results_frequency_dir, results_stats_dir, graphs_frequency_dir
        ), axis=1, result_type='expand')

    rain_data

    # %%
    aep_per = frequency_table.AEP_1_in_x[:-1]
    aep_per.name = 'AEP (1 in x)'
    aep_per

    # %%
    cl_mean = pd.DataFrame(rain_data.cl_mean_based.tolist())
    cl_mean = pd.concat(objs=[aep_per, cl_mean], axis=1).T
    cl_mean.columns = [ind for ind in rain_data.freq_label]
    cl_mean = pd.DataFrame(data=np.array(object=cl_mean.to_numpy(), dtype=np.float64), index=cl_mean.index, columns=cl_mean.columns)
    cl_mean.to_parquet(path='{}max_outflows_mean_based.parquet'.format(output_dir))

    cl_mean

    # %%
    cl_median = pd.DataFrame(rain_data.cl_median_based.tolist())
    cl_median = pd.concat(objs=[aep_per, cl_median], axis=1).T
    cl_median.columns = [ind for ind in rain_data.freq_label]
    cl_median = pd.DataFrame(data=np.array(object=cl_median.to_numpy(), dtype=np.float64), index=cl_median.index, columns=cl_median.columns)
    cl_median.to_parquet(path='{}max_outflows_median_based.parquet'.format(output_dir))

    cl_median

    # %%
    fig, ax = plt.subplots(figsize=(15,10), num=1, clear=True)

    matplotlib.use(backend='agg')

    ax.plot(
        cl_mean.loc['AEP (1 in x)'].to_numpy(),
        cl_mean.loc['mean'].to_numpy(),
        '-o',
        markersize=3,
        color='tab:blue',
        label='mean'
        )

    ax.plot(
        cl_mean.loc['AEP (1 in x)'].to_numpy(),
        cl_mean.loc['50%'].to_numpy(),
        '-.',
        linewidth=0.9,
        color='tab:red',
        label='median (50%)'
        )

    ax.fill_between(
        x=cl_mean.loc['AEP (1 in x)'].to_numpy(),
        y1=cl_mean.loc['5%'].to_numpy(),
        y2=cl_mean.loc['95%'].to_numpy(),
        alpha=0.3,
        color='tab:blue',
        label='conf. limits (5-95%)'
        )

    for ind1, ind2, ind3 in zip(
        cl_mean.columns.to_numpy(),
        cl_mean.loc['AEP (1 in x)'].to_numpy(), 
        cl_mean.loc['mean'].to_numpy()):
        ax.annotate(
            text=ind1,
            xy=(ind2, ind3),
            xytext=(5, -7.5),
            textcoords='offset points',
            fontsize=8
            )

    table = ax.table(
        cellText=cl_mean.to_numpy().round(decimals=3),
        rowLabels=cl_mean.index.to_numpy(),
        colLabels=cl_mean.columns.to_numpy(),
        bbox=[0,-0.62,1,0.5])
    table.set_fontsize(8)

    ax.legend()
    ax.grid(visible=True, which='both')
    ax.set_xscale(value='log')
    ax.set_title(label='Flow Quantiles - Mean-based', fontdict={'fontsize':10})
    ax.set_xlabel(xlabel='AEP (1 in x)', fontdict={'fontsize':9})
    ax.set_ylabel(ylabel='Flow Rate ($m^3/s$)', fontdict={'fontsize':9})

    fig.tight_layout()
    fig.savefig(fname='{}graph_max_outflows_mean_based.png'.format(output_dir), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig=fig)

    # %%
    fig, ax = plt.subplots(figsize=(15,10), num=1, clear=True)

    matplotlib.use(backend='agg')

    ax.plot(
        cl_median.loc['AEP (1 in x)'].to_numpy(),
        cl_median.loc['mean'].to_numpy(),
        '-o',
        markersize=3,
        color='tab:blue',
        label='mean'
        )

    ax.plot(
        cl_median.loc['AEP (1 in x)'].to_numpy(),
        cl_median.loc['50%'].to_numpy(),
        '-.',
        linewidth=0.9,
        color='tab:red',
        label='median (50%)'
        )

    ax.fill_between(
        x=cl_median.loc['AEP (1 in x)'].to_numpy(),
        y1=cl_median.loc['5%'].to_numpy(),
        y2=cl_median.loc['95%'].to_numpy(),
        alpha=0.3,
        color='tab:blue',
        label='conf. limits (5-95%)'
        )

    for ind1, ind2, ind3 in zip(
        cl_median.columns.to_numpy(),
        cl_median.loc['AEP (1 in x)'].to_numpy(), 
        cl_median.loc['mean'].to_numpy()):
        ax.annotate(
            text=ind1,
            xy=(ind2, ind3),
            xytext=(5, -7.5),
            textcoords='offset points',
            fontsize=8
            )

    table = ax.table(
        cellText=cl_median.to_numpy().round(decimals=3),
        rowLabels=cl_median.index.to_numpy(),
        colLabels=cl_median.columns.to_numpy(),
        bbox=[0,-0.62,1,0.5])
    table.set_fontsize(8)

    ax.legend()
    ax.grid(visible=True, which='both')
    ax.set_xscale(value='log')
    ax.set_title(label='Flow Quantiles - Median-based', fontdict={'fontsize':10})
    ax.set_xlabel(xlabel='AEP (1 in x)', fontdict={'fontsize':9})
    ax.set_ylabel(ylabel='Flow Rate ($m^3/s$)', fontdict={'fontsize':9})

    fig.tight_layout()
    fig.savefig(fname='{}graph_max_outflows_median_based.png'.format(output_dir), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig=fig)

# %%
if __name__ == '__main__':
    main()
