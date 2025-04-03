# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# %%
def main():

    # %%
    ifd_arr87 = './Assignment_2__Input_FE/inputs_2025_Autumn_PRM/ARR87_IFDs.csv'
    output_dir = './Assignment_2__Output_FE_PRM/'

    catch_area = 2.4

    C_10__min_contour = 0.6
    C_10__max_contour = 0.8
    C_10__mean_contour = 0.7

    num_simulations = 1_000_000

    zone = 'B' # 'A', 'B', 'C', 'D', 'E', 'F'
    elevation = 'Below 500m' # 'Above 500m', 'Below 500m'

    print(
        ifd_arr87,
        output_dir,
        catch_area,
        C_10__min_contour,
        C_10__max_contour,
        C_10__mean_contour,
        zone,
        elevation,
        sep='\n')

    # %%
    def create_output_dir(arg_output_dir):
        """create output directory if it does not exist

        arguments:
            arg_output_dir = [string] './Outputs_CM_Assignment_2/'
        """
        if not os.path.exists(arg_output_dir):
            os.makedirs(arg_output_dir)

    # %%
    create_output_dir(output_dir)

    # %%
    df_ifd = pd.read_csv(filepath_or_buffer=ifd_arr87, skiprows=2)
    df_ifd['duration'] = df_ifd.DURATION.str.split(pat=' ').str[0].astype(int)
    df_ifd['units'] = df_ifd.DURATION.str.split(pat=' ').str[1]
    df_ifd.units = df_ifd.units.map(arg={'mins':'min', 'hours':'hr', 'hour':'hr'})
    df_ifd.duration = df_ifd.apply(func=lambda arg: pd.to_timedelta(arg=arg.duration, unit=arg.units), axis=1)
    df_ifd.duration = df_ifd.duration / pd.Timedelta(minutes=1)
    df_ifd['duration'] = df_ifd['duration'].astype(int)
    df_ifd.drop(columns=['DURATION', 'units'], inplace=True)
    df_ifd.set_index(keys='duration', inplace=True)
    df_ifd.columns = df_ifd.columns.astype(int)

    df_ifd

    # %%
    tc = 0.76*(catch_area**0.38)
    tc = tc*60

    tc

    # %%
    # Find the insertion point
    idx = np.searchsorted(a=df_ifd.index.to_numpy(), v=tc)

    # Determine the limits
    if idx == 0:
        lower_limit = None
        upper_limit = df_ifd.index.to_numpy()[0]
    elif idx == len(df_ifd.index.to_numpy()):
        lower_limit = df_ifd.index.to_numpy()[-1]
        upper_limit = None
    else:
        lower_limit = df_ifd.index.to_numpy()[idx - 1]
        upper_limit = df_ifd.index.to_numpy()[idx]

    idx = np.array(object=[lower_limit, upper_limit])
    idx

    # %%
    I_minh_Xy = df_ifd.loc[idx.min(),:].to_numpy()
    I_maxh_Xy = df_ifd.loc[idx.max(),:].to_numpy()

    I_minh_Xy, I_maxh_Xy

    # %%
    # I_1h_100y = df_ifd.loc[60,100]
    # I_2h_100y = df_ifd.loc[120,100]

    # I_1h_100y, I_2h_100y

    # %%
    I_catch = [np.interp(x=tc, xp=[60,120], fp=[ind1, ind2]) for ind1,ind2 in zip(I_minh_Xy, I_maxh_Xy)]
    I_catch = np.array(object=I_catch)

    I_catch

    # %%
    # I_catch = np.interp(x=tc, xp=[60,120], fp=[I_1h_100y,I_2h_100y])

    # I_catch

    # %%
    # Mean and standard deviation
    contours = np.array(object=[C_10__min_contour, C_10__max_contour])
    six_sigma = 2*np.min(a=abs(contours - C_10__mean_contour))
    std_dev = six_sigma / 6
    contours = np.array(object=[C_10__mean_contour - 3*std_dev, C_10__mean_contour + 3*std_dev])

    # Generate random points with a normal distribution
    C_10 = np.random.normal(C_10__mean_contour, std_dev, num_simulations)

    # Filter points to be within the desired range (0.7 to 0.8)
    C_10 = C_10[(C_10 >= contours.min()) & (C_10 <= contours.max())]
    C_10 = np.sort(a=C_10)

    C_10

    # %%
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        x=C_10,
        bins=100,
        color='tab:blue',
        alpha=0.7,
        edgecolor='black',
        linewidth=1.2
        );
    ax.grid(visible=True)
    ax.set_title(label=r'Histogram of C$_{10}$ values')
    ax.set_xlabel(xlabel=r'C$_{10}$ value')
    ax.set_ylabel(ylabel='Frequency')

    fig.savefig(
        os.path.join(output_dir, 'C_10__histogram.png'),
        dpi=300,
        bbox_inches='tight'
    )

    # %%
    I_12h_50y = df_ifd.loc[720,50]
    I_12h_2y = df_ifd.loc[720,2]

    I_12h_50y, I_12h_2y

    # %%
    constant_50 = 0.366*I_12h_50y/I_12h_2y
    constant_100 = 0.588*I_12h_50y/I_12h_2y

    constant_50, constant_100

    # %%
    if (zone == 'A') & (elevation == 'Below 500m'):
        FF_y = {
            'FF_1': 0.67, 'FF_2': 0.81, 'FF_5': 0.92, 'FF_10': 1, 'FF_20': 1.07, 
            'FF_50': 1.90 - constant_50, 
            'FF_100': 2.45 - constant_100,
            }
    elif (zone == 'A') & (elevation == 'Above 500m'):
        FF_y = {
            'FF_1': 0.50, 'FF_2': 0.66, 'FF_5': 0.85, 'FF_10': 1, 'FF_20': 1.14, 
            'FF_50': 1.32, 
            'FF_100': 1.46,
            }
    elif (zone == 'B') & (elevation == 'Below 500m'):
        FF_y = {
            'FF_1': 0.62, 'FF_2': 0.74, 'FF_5': 0.88, 'FF_10': 1, 'FF_20': 1.12, 
            'FF_50': 1.99 - constant_50, 
            'FF_100': 2.57 - constant_100,
            }
    elif (zone == 'B') & (elevation == 'Above 500m'):
        FF_y = {
            'FF_1': 0.57, 'FF_2': 0.70, 'FF_5': 0.86, 'FF_10': 1, 'FF_20': 1.14, 
            'FF_50': 1.33, 
            'FF_100': 1.50,
            }
    elif (zone == 'C') & (elevation == 'Below 500m'):
        FF_y = {
            'FF_1': 0.62, 'FF_2': 0.78, 'FF_5': 0.90, 'FF_10': 1, 'FF_20': 1.10, 
            'FF_50': 1.97 - constant_50, 
            'FF_100': 2.54 - constant_100,
            }
    elif (zone == 'C') & (elevation == 'Above 500m'):
        FF_y = {
            'FF_1': 0.89, 'FF_2': 0.92, 'FF_5': 0.95, 'FF_10': 1, 'FF_20': 1.05, 
            'FF_50': 1.17, 
            'FF_100': 1.24,
            }
    elif (zone == 'D') & (elevation == 'Below 500m'):
        FF_y = {
            'FF_1': 0.43, 'FF_2': 0.58, 'FF_5': 0.80, 'FF_10': 1, 'FF_20': 1.20, 
            'FF_50': 1.54, 
            'FF_100': 1.80,
            }
    elif (zone == 'D') & (elevation == 'Above 500m'):
        FF_y = {
            'FF_1': 0.37, 'FF_2': 0.53, 'FF_5': 0.77, 'FF_10': 1, 'FF_20': 1.25, 
            'FF_50': 1.74, 
            'FF_100': 2.20,
            }
    elif (zone == 'E') & (elevation == 'Below 500m'):
        FF_y = {
            'FF_1': 0.38, 'FF_2': 0.54, 'FF_5': 0.78, 'FF_10': 1, 'FF_20': 1.26, 
            'FF_50': 1.71, 
            'FF_100': 2.14,
            }
    elif (zone == 'E') & (elevation == 'Above 500m'):
        FF_y = {
            'FF_1': 0.52, 'FF_2': 0.64, 'FF_5': 0.82, 'FF_10': 1, 'FF_20': 1.21, 
            'FF_50': 1.52, 
            'FF_100': 1.78,
            }
    elif (zone == 'F') & (elevation == 'Below 500m'):
        FF_y = {
            'FF_1': 0.66, 'FF_2': 0.74, 'FF_5': 0.87, 'FF_10': 1, 'FF_20': 1.15, 
            'FF_50': 1.39, 
            'FF_100': 1.60,
            }
    elif (zone == 'F') & (elevation == 'Above 500m'):
        FF_y = {
            'FF_1': 0.69, 'FF_2': 0.77, 'FF_5': 0.89, 'FF_10': 1, 'FF_20': 1.10, 
            'FF_50': 1.26, 
            'FF_100': 1.34,
            }
    else:
        print('Zone and elevation not found')

    FF_y

    # %%
    # FF_50 = 1.99 - 0.366*I_12h_50y/I_12h_2y
    # FF_100 = 2.57 - 0.588*I_12h_50y/I_12h_2y

    # FF_50, FF_100

    # %%
    # FF_y = {
    #     'FF_1': 0.62, 
    #     'FF_2': 0.74, 
    #     'FF_5': 0.88, 
    #     'FF_10': 1, 
    #     'FF_20': 1.12,
    #     'FF_50': FF_50,
    #     'FF_100': FF_100,
    #     }

    # FF_y

    # %%
    C_y = np.array(object=[C_10*ind for ind in np.array(object=list(FF_y.values()))])

    C_y

    # %%
    C_y.shape

    # %%
    C_y__stats = pd.DataFrame(data=np.transpose(a=C_y), columns=df_ifd.columns)
    C_y__stats = C_y__stats.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    C_y__stats.to_csv(
        path_or_buf=os.path.join(output_dir, 'C_y__stats.csv'),
        index=True,
        header=True
    )

    C_y__stats

    # %%
    x = C_y__stats.columns.to_numpy()
    y_mean = C_y__stats.loc['mean'].to_numpy()
    y_05 = C_y__stats.loc['5%'].to_numpy()
    y_25 = C_y__stats.loc['25%'].to_numpy()
    y_50 = C_y__stats.loc['50%'].to_numpy()
    y_75 = C_y__stats.loc['75%'].to_numpy()
    y_95 = C_y__stats.loc['95%'].to_numpy()
    y_min = C_y__stats.loc['min'].to_numpy()
    y_max = C_y__stats.loc['max'].to_numpy()

    # %%
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y_mean, color='tab:blue', label='Mean', alpha=1)
    ax.plot(x, y_50, color='tab:blue', label='Median', alpha=0.3)
    ax.fill_between(x=x, y1=y_min, y2=y_max, color='tab:blue', alpha=0.3)
    ax.fill_between(x=x, y1=y_05, y2=y_95, color='tab:blue', alpha=0.3)
    ax.fill_between(x=x, y1=y_25, y2=y_75, color='tab:blue', alpha=0.3)

    for ind1, ind2 in zip(x, y_mean):
        ax.annotate(
            text='{}: {:.2f}'.format(int(ind1), ind2),
            xy=(ind1, ind2),
            xytext=(5, 0),
            textcoords='offset points',
            va='center',
            ha='left',
            fontsize=8
            )

    for ind1, ind2 in zip(x, y_95):
        ax.annotate(
            text='{}: {:.2f}'.format(int(ind1), ind2),
            xy=(ind1, ind2),
            xytext=(5, 7),
            textcoords='offset points',
            va='center',
            ha='left',
            fontsize=8
            )

    for ind1, ind2 in zip(x, y_05):
        ax.annotate(
            text='{}: {:.2f}'.format(int(ind1), ind2),
            xy=(ind1, ind2),
            xytext=(5, -7),
            textcoords='offset points',
            va='center',
            ha='left',
            fontsize=8
            )

    ax.set_xscale(value='log')
    ax.grid(which='both', axis='x')
    ax.grid(axis='y')

    ax.set_title(label=r'C$_y$ distribution across ARI')
    ax.set_xlabel(xlabel='ARI (years)')
    ax.set_ylabel(ylabel=r'C$_y$ value')

    fig.savefig(
        os.path.join(output_dir, 'C_y__stats.png'),
        dpi=300,
        bbox_inches='tight'
    )

    # %%
    # C100 = FF_100 * C10_value
    # C100

    # %%
    Q_y = C_y*np.reshape(a=I_catch, newshape=(7,1))*catch_area/3.6

    Q_y

    # %%
    Q_y__stats = pd.DataFrame(data=np.transpose(a=Q_y), columns=df_ifd.columns)
    Q_y__stats = Q_y__stats.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    Q_y__stats.to_csv(
        path_or_buf=os.path.join(output_dir, 'Q_y__stats.csv'),
        index=True,
        header=True
    )

    Q_y__stats

    # %%
    x = Q_y__stats.columns.to_numpy()
    y_mean = Q_y__stats.loc['mean'].to_numpy()
    y_05 = Q_y__stats.loc['5%'].to_numpy()
    y_25 = Q_y__stats.loc['25%'].to_numpy()
    y_50 = Q_y__stats.loc['50%'].to_numpy()
    y_75 = Q_y__stats.loc['75%'].to_numpy()
    y_95 = Q_y__stats.loc['95%'].to_numpy()
    y_min = Q_y__stats.loc['min'].to_numpy()
    y_max = Q_y__stats.loc['max'].to_numpy()

    # %%
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y_mean, color='tab:blue', label='Mean', alpha=1)
    ax.plot(x, y_50, color='tab:blue', label='Median', alpha=0.3)
    ax.fill_between(x=x, y1=y_min, y2=y_max, color='tab:blue', alpha=0.3)
    ax.fill_between(x=x, y1=y_05, y2=y_95, color='tab:blue', alpha=0.3)
    ax.fill_between(x=x, y1=y_25, y2=y_75, color='tab:blue', alpha=0.3)

    for ind1, ind2 in zip(x, y_mean):
        ax.annotate(
            text='{}: {:.2f}'.format(int(ind1), ind2),
            xy=(ind1, ind2),
            xytext=(5, 0),
            textcoords='offset points',
            va='center',
            ha='left',
            fontsize=8
            )

    for ind1, ind2 in zip(x, y_95):
        ax.annotate(
            text='{}: {:.2f}'.format(int(ind1), ind2),
            xy=(ind1, ind2),
            xytext=(5, 7),
            textcoords='offset points',
            va='center',
            ha='left',
            fontsize=8
            )

    for ind1, ind2 in zip(x, y_05):
        ax.annotate(
            text='{}: {:.2f}'.format(int(ind1), ind2),
            xy=(ind1, ind2),
            xytext=(5, -7),
            textcoords='offset points',
            va='center',
            ha='left',
            fontsize=8
            )

    ax.set_xscale(value='log')
    ax.grid(which='both', axis='x')
    ax.grid(axis='y')

    ax.set_title(label=r'Q$_y$ distribution across ARI')
    ax.set_xlabel(xlabel='ARI (years)')
    ax.set_ylabel(ylabel=r'Q$_y$ value (m$^3$/s)')
    fig.savefig(
        os.path.join(output_dir, 'Q_y__stats.png'),
        dpi=300,
        bbox_inches='tight'
    )

    # %%
    FF_y

    # %%
    _inputs = {
        'Catchment Area (km^2)': catch_area,
        'C_10 min contour': C_10__min_contour,
        'C_10 max contour': C_10__max_contour,
        'C_10 mean contour': C_10__mean_contour,
        'Zone': zone,
        'Elevation': elevation,
        'tc (min)': tc,
        'tc (h)': tc/60,
        'I_12h_50y (mm)': I_12h_50y,
        'I_12h_2y (mm)': I_12h_2y,
        }
    _inputs = pd.DataFrame(data=list(_inputs.values()), index=_inputs.keys(), columns=['Values'])
    _inputs.index.name = 'Input Parameters'
    _inputs.to_csv(
        path_or_buf=os.path.join(output_dir, '_inputs.csv'),
        index=True,
        header=True
    )

    _inputs

    # %%
    _calculations = np.vstack((
        I_minh_Xy, I_maxh_Xy, I_catch, 
        np.array(object=list(FF_y.values())),
        C_y__stats.loc['mean'].to_numpy(),
        Q_y__stats.loc['mean'].to_numpy(),
        ))
    _calculations = pd.DataFrame(data=_calculations, index=['I_minh_Xy', 'I_maxh_Xy', 'I_catch', 'FF_y', 'C_y__mean', 'Q_y__mean'], columns=df_ifd.columns)
    _calculations.to_csv(
        path_or_buf=os.path.join(output_dir, '_calculations.csv'),
        index=True,
        header=True
    )

    _calculations

# %%
if __name__ == '__main__':
    main()


