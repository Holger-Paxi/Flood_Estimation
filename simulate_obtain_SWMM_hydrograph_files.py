# %% [markdown]
# # import python packages

# %%
import pandas as pd
import os
from pyswmm import Simulation, Links

# %% [markdown]
# # get hydrographs from link

# %%
Link_ID = '116'

# %% [markdown]
# # read SWMM filenames

# %%
swmm_files = []

for file in os.listdir('../Output_SWMM_directory/'):
    if file.endswith('.inp'):
        swmm_files.append(file)

swmm_files = pd.DataFrame(swmm_files, columns=['swmm_filename'])
swmm_files.sort_values(by='swmm_filename', inplace=True, ignore_index=True)

# %% [markdown]
# # simulate SWMM files, export SWMM results

# %%

for index_1 in swmm_files.index:
    a1 = swmm_files.swmm_filename[index_1]
    with Simulation(r'../Output_SWMM_directory/' + a1) as sim:
        Link_swmm = Links(sim)[Link_ID]
        temp_list = []
        for index_2, step in enumerate(sim):
            a2 = pd.DataFrame({'timestamp':[sim.current_time], 'flow':[Link_swmm.flow]})
            temp_list.append(a2)
        temp_list = pd.concat(temp_list, ignore_index=True)
        temp_list.to_parquet('../Output_SWMM_directory/' + a1[:-4] + '.parquet')
