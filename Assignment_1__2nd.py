# %%
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# %%
def main():

    # %%
    input_dir = './Assignment_1__Output_FE__Flike/'
    output_dir = './Assignment_1__Output_FE__2nd/'
    arg_skip_row_ind_GEV = {'c1':[100,99,99], 'c2':[98,98,98], 'c3':[99,98,98], 'full':[128]}
    arg_skip_row_ind_LP3 = {'c1':[100,100,100], 'c2':[98,98,98], 'c3':[99,98,99], 'full':[137]}

    print(
        input_dir,
        output_dir,
        arg_skip_row_ind_GEV,
        arg_skip_row_ind_LP3,
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
    def list_files(arg_directory_path, arg_regex, arg_column_name='file_path'):
        """return list of files in a directory

        arguments:
            [string] --> arg_directory_path = directory path of the polygons
            [string] --> arg_regex = regex entry
            [string] --> arg_column_name = column's name
        """
        list_files = glob.glob(pathname=arg_directory_path + arg_regex)
        list_files = pd.DataFrame(list_files, columns=[arg_column_name])
        list_files.sort_values(by=[arg_column_name], inplace=True)
        list_files.reset_index(drop=True, inplace=True)

        return list_files

    # %%
    arg_skip_row_ind_GEV = list(ind2 for ind1 in arg_skip_row_ind_GEV.values() for ind2 in ind1)
    arg_skip_row_ind_LP3 = list(ind2 for ind1 in arg_skip_row_ind_LP3.values() for ind2 in ind1)

    print(arg_skip_row_ind_LP3, arg_skip_row_ind_GEV)

    # %%
    results = list_files(input_dir, '*.txt', 'file_name')
    results.file_name = results.file_name.str.split(pat='/').str[-1]
    results['data_case'] = results.file_name.str.split(pat='__').str[1]
    results['prob_fit'] = results.file_name.str.split(pat='__').str[-1].str.split(pat='.').str[0]
    results.sort_values(by=['prob_fit', 'data_case'], inplace=True)
    results.reset_index(drop=True, inplace=True)
    results = pd.concat(objs=[results, pd.Series(data=arg_skip_row_ind_GEV + arg_skip_row_ind_LP3, name='skip_row_ind')], axis=1)
    results

    # %%
    def func_read_table(arg_input_dir, arg_output_dir, arg_file_name, arg_data_case, arg_prob_fit, arg_skip_row_ind):
        df = pd.read_table(
            filepath_or_buffer='{}{}'.format(arg_input_dir, arg_file_name), 
            sep='\s+', header=None, names=['ARI', 'flow_rate', 'lower_5', 'upper_95'], 
            skiprows=arg_skip_row_ind, nrows=12
            )
        df['diff_perc_lower'] = 100*df.lower_5/df.flow_rate
        df['diff_perc_upper'] = 100*df.upper_95/df.flow_rate
        df.to_csv(path_or_buf='{}{}_{}.csv'.format(arg_output_dir, arg_prob_fit, arg_data_case), index=False)

        return df

    # %%
    results['df'] = results.apply(func=lambda arg: func_read_table(input_dir, output_dir, arg.file_name, arg.data_case, arg.prob_fit, arg.skip_row_ind), axis=1)
    results

    # %%
    def prepare_data(arg_output_dir, arg_prob_fit, arg_data_case, arg_df):
        df = arg_df.iloc[[5,7,8,9,10,11]].copy()
        df.reset_index(drop=True, inplace=True)
        df.to_csv(path_or_buf='{}df_{}_{}.csv'.format(arg_output_dir, arg_prob_fit, arg_data_case), index=False)
        return df

    # %%
    results['df_data'] = results.apply(func=lambda arg: prepare_data(output_dir, arg.prob_fit, arg.data_case, arg.df), axis=1)
    results

    # %%
    def plot_ffa(arg_output_dir, arg_prob_fit, arg_data_case, arg_df, arg_df_data):

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(
            arg_df.ARI.to_numpy(),
            arg_df.flow_rate.to_numpy(),
            '-o',
            color='tab:blue',
            linewidth=0.5,
            markersize=3
            )

        ax.fill_between(
            x=arg_df.ARI.to_numpy(),
            y1=arg_df.upper_95.to_numpy(),
            y2=arg_df.lower_5.to_numpy(),
            color='tab:blue',
            alpha=0.2
            )

        for ind1, ind2 in zip(arg_df_data.ARI.to_numpy(), arg_df_data.flow_rate.to_numpy()):
            ax.annotate(
                text='{}: {}'.format(int(ind1), ind2),
                xy=(ind1, ind2),
                xytext=(5, 0),
                textcoords='offset points',
                va='center',
                ha='left',
                fontsize=8
                )

        for ind1, ind2 in zip(arg_df_data.ARI.to_numpy(), arg_df_data.upper_95.to_numpy()):
            ax.annotate(
                text='{}: {}'.format(int(ind1), ind2),
                xy=(ind1, ind2),
                xytext=(5, 7),
                textcoords='offset points',
                va='center',
                ha='left',
                fontsize=8
                )

        for ind1, ind2 in zip(arg_df_data.ARI.to_numpy(), arg_df_data.lower_5.to_numpy()):
            ax.annotate(
                text='{}: {}'.format(int(ind1), ind2),
                xy=(ind1, ind2),
                xytext=(5, -7),
                textcoords='offset points',
                va='center',
                ha='left',
                fontsize=8
                )

        ax.grid(visible=True, which='both')
        ax.set_xscale(value='log')
        ax.set_title(label='Flood Flow Rate vs Annual Recurrence Interval\n{}_{}'.format(arg_prob_fit, arg_data_case))
        ax.set_xlabel(xlabel='Annual Recurrence Interval in ($year$)')
        ax.set_ylabel(ylabel='Flow Rate in ($m^3/s$)')
        fig.savefig(fname='{}plot_{}_{}.png'.format(arg_output_dir, arg_prob_fit, arg_data_case), bbox_inches='tight')

        # close figure
        plt.cla()
        plt.clf()
        plt.close(fig=fig)

    # %%
    results.apply(func=lambda arg: plot_ffa(output_dir, arg.prob_fit, arg.data_case, arg.df, arg.df_data), axis=1)
    results

    # %%
    results_GEV_full, results_LP3_full = results.iloc[:10,:], results.iloc[10:,:]
    results_LP3_full.reset_index(drop=True, inplace=True)

    results_GEV_full = [results_GEV_full.iloc[9:,:], results_GEV_full.iloc[:3,:], results_GEV_full.iloc[3:6,:], results_GEV_full.iloc[6:9,:]]
    results_LP3_full = [results_LP3_full.iloc[9:,:], results_LP3_full.iloc[:3,:], results_LP3_full.iloc[3:6,:], results_LP3_full.iloc[6:9,:]]

    results_GEV_full = [ind.reset_index(drop=True) for ind in results_GEV_full]
    results_LP3_full = [ind.reset_index(drop=True) for ind in results_LP3_full]

    results_GEV_full, results_GEV_c1, results_GEV_c2, results_GEV_c3 = results_GEV_full
    results_LP3_full, results_LP3_c1, results_LP3_c2, results_LP3_c3 = results_LP3_full

    print(results_GEV_full, results_GEV_c1, results_GEV_c2, results_GEV_c3, results_LP3_full, results_LP3_c1, results_LP3_c2, results_LP3_c3, sep='\n')

    # %%
    results_GEV = [results_GEV_c1, results_GEV_c2, results_GEV_c3]
    results_LP3 = [results_LP3_c1, results_LP3_c2, results_LP3_c3]

    # %%
    def plot_ffa_comparison(arg_output_dir, arg_df_full, arg_df_c1, arg_df_c2, arg_df_c3):

        data = [arg_df_full, arg_df_c1, arg_df_c2, arg_df_c3]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        count = np.arange(stop=len(data))

        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8,16))

        for key, value, color in zip(count, data, colors):
            ax[key].plot(
                value.df.ARI.to_numpy(),
                value.df.flow_rate.to_numpy(),
                '-o',
                color=color,
                linewidth=0.5,
                markersize=2
                )

            ax[key].fill_between(
                x=value.df.ARI.to_numpy(),
                y1=value.df.upper_95.to_numpy(),
                y2=value.df.lower_5.to_numpy(),
                color=color,
                alpha=0.2
                )

            for ind1, ind2 in zip(value.df_data.ARI.to_numpy(), value.df_data.flow_rate.to_numpy()):
                ax[key].annotate(
                    text='{}: {}'.format(int(ind1), ind2),
                    xy=(ind1, ind2),
                    xytext=(5, 0),
                    textcoords='offset points',
                    va='center',
                    ha='left',
                    fontsize=8
                    )

            for ind1, ind2 in zip(value.df_data.ARI.to_numpy(), value.df_data.upper_95.to_numpy()):
                ax[key].annotate(
                    text='{}: {}'.format(int(ind1), ind2),
                    xy=(ind1, ind2),
                    xytext=(5, 7),
                    textcoords='offset points',
                    va='center',
                    ha='left',
                    fontsize=8
                    )

            for ind1, ind2 in zip(value.df_data.ARI.to_numpy(), value.df_data.lower_5.to_numpy()):
                ax[key].annotate(
                    text='{}: {}'.format(int(ind1), ind2),
                    xy=(ind1, ind2),
                    xytext=(5, -7),
                    textcoords='offset points',
                    va='center',
                    ha='left',
                    fontsize=8
                    )

            ax[key].grid(visible=True, which='both')
            ax[key].set_xscale(value='log')
            ax[key].set_title(label='{}_{}'.format(value.prob_fit, value.data_case), loc='left', fontdict={'fontsize':10})

        fig.suptitle(
            t='Flood Flow Rate vs Annual Recurrence Interval\n{} Probability Model - {}'.format(
                value.prob_fit, 
                value.data_case.rsplit(sep='_', maxsplit=3)[0]
                ), 
            x=0.5, 
            y=0.91
            )
        fig.supxlabel(t='Annual Recurrence Interval in ($year$)', x=0.5, y=0.08)
        fig.supylabel(t='Flow Rate in ($m^3/s$)', x=0.03, y=0.5)

        fig.savefig(
            fname='{}all_{}_plots_{}.png'.format(
                arg_output_dir, 
                value.prob_fit, 
                value.data_case.rsplit(sep='_', maxsplit=3)[0]
                ),
            bbox_inches='tight'
            )

    # %%
    for ind1 in results_GEV:
        var = [output_dir, results_GEV_full.iloc[0]] + [ind2[1] for ind2 in ind1.iterrows()]
        plot_ffa_comparison(*var)

    # %%
    for ind1 in results_LP3:
        var = [output_dir, results_LP3_full.iloc[0]] + [ind2[1] for ind2 in ind1.iterrows()]
        plot_ffa_comparison(*var)

    # %%
    df_GEV = []

    for ind in results_GEV:
        df = []
        for ind1, ind2, ind3 in zip(
            pd.concat(objs=[results_GEV_full.prob_fit, ind.prob_fit]), 
            pd.concat(objs=[results_GEV_full.data_case, ind.data_case]), 
            pd.concat(objs=[results_GEV_full.df_data, ind.df_data])
            ):
            ind3.index = pd.MultiIndex.from_arrays(arrays=[[ind1]*len(ind3), [ind2]*len(ind3)])
            df.append(ind3)
        df = pd.concat(objs=df)
        df.to_csv(path_or_buf='{}all_df_{}_{}.csv'.format(output_dir, ind1, ind2.rsplit(sep='_', maxsplit=3)[0]))
        df_GEV.append(df)

    df_GEV

    # %%
    df_LP3 = []

    for ind in results_LP3:
        df = []
        for ind1, ind2, ind3 in zip(
            pd.concat(objs=[results_LP3_full.prob_fit, ind.prob_fit]), 
            pd.concat(objs=[results_LP3_full.data_case, ind.data_case]), 
            pd.concat(objs=[results_LP3_full.df_data, ind.df_data])
            ):
            ind3.index = pd.MultiIndex.from_arrays(arrays=[[ind1]*len(ind3), [ind2]*len(ind3)])
            df.append(ind3)
        df = pd.concat(objs=df)
        df.to_csv(path_or_buf='{}all_df_{}_{}.csv'.format(output_dir, ind1, ind2.rsplit(sep='_', maxsplit=3)[0]))
        df_LP3.append(df)

    df_LP3

# %%
if __name__ == '__main__':
    main()


