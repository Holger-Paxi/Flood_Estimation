# Flood_Estimation

## generate_rainfall_temporal_patterns

- The `generate_rainfall_temporal_patterns.ipynb` and `generate_rainfall_temporal_patterns.py` scripts generate SWMM files from rainfall temporal patterns
- inputs:
    - from `Inputs_FE_Assignment_3` directory:
        - from `ECsouth` directory:
            - `ECsouth_AllStats.csv` file that contains the **burst duration**
            - `ECsouth_Increments.csv` file that contains the **temporal patterns**
        - from `depths_33.8774_151.093` folder:
            - `depths_-33.8774_151.093_all_design.csv` file that contains the **precipitation depths**
        - `971007_SW5.inp` file that contain the pre-configured SWMM file. The modiication is in the line '704'
- outputs:
    - the `Output_SWMM_directory` directory is created which contains all SWMM file outputs. The format of the SWMM file name is `{frequent-intermediate-rare}__{freq_AEP}__{duration: from 10 to 120 minutes}__{pattern: from 1 to 10}`
- other inputs to consider in the script:
    - `timestep` = 5
    - `starting_date` = dt.datetime(year=2023, month=1, day=1, hour=0, minute=5, second=0)
    - `rain_gauge` = 'RainSeries1'
    - `orig_SWMM_file` = '../Inputs_FE_Assignment_3/971007_SW5.inp'
    - `output_directory` = 'Output_SWMM_directory'
