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
    - the `Output_SWMM_directory` directory is created which contains all SWMM file outputs. The format of the SWMM file name is `{frequent-intermediate-rare}__{freq_AEP}__{duration: from 10 to 120 minutes}__{pattern: from 1 to 10}.inp`
- other input variables:
    - `timestep` = 5
    - `starting_date` = dt.datetime(year=2023, month=1, day=1, hour=0, minute=5, second=0)
    - `rain_gauge` = 'RainSeries1'
    - `orig_SWMM_file` = '../Inputs_FE_Assignment_3/971007_SW5.inp'
    - `output_directory` = 'Output_SWMM_directory'

## simulate_obtain_SWMM_hydrographs_files

- the `simulate_obtain_SWMM_hydrographs_files.ipynb` and `simulate_obtain_SWMM_hydrographs_files.py` scripts run simulations by using SWMM files (\*.inp), generate timeseries and store them in a parquet file (\*.parquet)
- inputs:
    - the SWMM files are simmulated are from the `Output_SWMM_directory` directory. The SWMM files has the following format: `{frequent-intermediate-rare}__{freq_AEP}__{duration: from 10 to 120 minutes}__{pattern: from 1 to 10}.inp`
- outputs:
    - the hydrograph timeseries are stored in `Output_SWMM_directory` following the same name format: `{frequent-intermediate-rare}__{freq_AEP}__{duration: from 10 to 120 minutes}__{pattern: from 1 to 10}.parquet`
- other input variables:
    - `Link_ID` = 116

## generate_stats_plot_SWMM_hydrograph

- the `generate_stats_plot_SWMM_hydrograph.ipynb` script does the statistical analysis from the SWMM results (parquet files)
- inputs:
    - from `Output_SWMM_directory` directory, SWMM reports in parquet extension. Their name formats are: `{frequent-intermediate-rare}__{freq_AEP}__{duration: from 10 to 120 minutes}__{pattern: from 1 to 10}.parquet`
- outputs:
    - `GSPSWMMH_max_flows.csv` file that contains the maximum flow for each hydrograph through all the SWMM simulations
    - `GSPSWMMH_statistics.csv` file that contains the statistical analysis of all the flows for each duration and frequency
    - `GSPSWMMH_design_flows.csv` file that contains the design flows, boundary limits, median and average for each frequency
    - `GSPSWMMH__{frequent-intermediate-rare}__{freq_AEP}.png` files that are box-plots for each frequency
    - `GSPSWMMH_design_flows.png` file that is a logarithmic plot that contains all the design flows, and boundary limits
     