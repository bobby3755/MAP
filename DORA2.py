import sma_lib.MAP_Parameters as params
import numpy as np
import pandas as pd
import os


xmlname = "DORA2_settings"

#read in the settings in the .xml file using hazen's Parameter Class
pars = params.Parameters(xmlname+'.xml') #par is an object of type Parameters, defined in sa_library
#to access parameters, use par.parameter name. eg par.start_frame
#note these values can be manually changed: par.frameset = 200 replaces whatever was there.
print(pars.end_frame)

def load_csv(selected_csv,dir_path,pars = pars, **kwargs):
    '''Organizes data frame by trimming frames to start and endframe and removing invalid readings'''
    # read CSV file into Pandas DataFrame
    csv_path = os.path.join(dir_path, selected_csv)
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Add an index column as the first column
    df.insert(0, 'index', range(len(df)))

    # Rename the first four columns as 'index', 'X position', 'Y position', and 'Intensity'
    df.columns.values[:4] = ['index','X position', 'Y position', 'Intensity']

    # Display the updated DataFrame
    print(df)

    pre_data = df

    
    #load in kwargs
    start_frame = kwargs.get('start_frame', pars.start_frame)
    end_frame = kwargs.get('end_frame', pars.end_frame)
    time_step = kwargs.get('time_step',pars.time_step)

    #section data from frame start to frame end
    pre_data = pre_data.iloc[start_frame:end_frame]

    # create a boolean array of where 1s are when x position is 0 or invalid
    # this is bc Ryan's code exports invalid readings as (0,0)
    ind_invalid_reading = pre_data['X position'] == 0

    # SEPARATE data into front and back end (front==graphing ; back == tables)
    # if the index is not invalid (or valid) keep it and store in data
    data = pre_data[~ind_invalid_reading].copy()
    # section the pre data for all the invalid values
    data_back = pre_data[ind_invalid_reading].copy()

    # in data back develop a time colomn
    data_back['Time (ms)'] = data_back['index']*time_step

    # set all target x positions to NaN, if the reading was excluded due to invalid reading
    data_back['X position'] = np.nan
    # set all target y positions to NaN, if the reading was excluded due to invalid reading
    data_back['Y position'] = np.nan
    data_back['Excluded Type'] = 'Invalid Reading'

    print(f" The following CSV has been read: {selected_csv}")

    return data, ind_invalid_reading, data_back

def find_center(data, pars = pars,**kwargs):

    #load kwargs
    centering_strategy = kwargs.get('centering_strategy',pars.centering_strategy)

    #center
    if centering_strategy == "center_hist_max":
        
        #load kwargs
        bin_num = kwargs.get('bin_num',pars.center_hist_max_bins)

        #establish values
        x = data["X position"]
        y = data["Y position"]

        # find the center of a plot using a low resolution histogram. Find the max value of that histogram. The corresponding x and y indicies become those of the center make the center
        H, xedges , yedges = np.histogram2d(x,y, bins = bin_num)

        #find the x and y index of the maximum value histogram 
        indMax = np.unravel_index(H.argmax(),H.shape)

        #Set the value of the max x and y indexes to be the new OR(OverRidden) center
        center = [xedges[indMax[0]],yedges[indMax[1]]]
        radius_estimate = None
        print(f"From {centering_strategy}, the center is {center}")
   
    
    if centering_strategy == "center_circular_trajectory":

        total_guesses = 10000
        n = 2
        num_points = int(total_guesses**(1/n))

        # find uniform guesses in range of max and min unaltered data values for y position
        guess_y = np.linspace(data.iloc[:, 2].max(), data.iloc[:, 2].min(), num_points)

        # find guesses for x position
        guess_x = np.linspace(data.iloc[:, 1].max(), data.iloc[:, 1].min(), num_points)

        # permute each x and y center guess together to create 10,000 unique center guesses
        center_x, center_y = np.meshgrid(guess_x, guess_y)
        center_guesses = np.column_stack((center_x.ravel(), center_y.ravel()))

        # calculate distances between each point in the data and all center guesses
        distances = np.sqrt((np.array(data['X position'])[:, np.newaxis] - center_guesses[:, 0])**2 + 
                            (np.array(data['Y position'])[:, np.newaxis] - center_guesses[:, 1])**2)

        # calculate average distances (radii) for each center guess
        ave_distance = np.mean(distances, axis=0)

        # calculate standard deviation of distances to each point in the trajectory for each center guess
        std = np.std(distances, axis=0)

        # find the index of the center guess with the lowest std
        target_row = np.argmin(std)

        # retrieve the center coordinates and radius for the best guess
        center_x = center_guesses[target_row, 0]
        center_y = center_guesses[target_row, 1]
        radius_estimate = ave_distance[target_row]
        center = (center_x, center_y)
        print(f"From {centering_strategy}, the center is {center} with a radius of {radius_estimate}")

    return center, radius_estimate

def calculate_time_angle(data,center,pars=pars,**kwargs):

    #unload pixel size
    pixel_size = kwargs.get('pixel_size',pars.pixel_size)
    time_step = kwargs.get('time_step',pars.time_step)

    ## Step 1: center the data
    # substract averages from each column to find displacement, store into new columns
    data["X displacement (pixels)"] = data['X position'] - center[0]
    data["Y displacement (pixels)"] = data['Y position'] - center[1]
    ## Step 2: Convert pixels to nm
    data["X displacement (nm)"] = data['X displacement (pixels)']*pixel_size
    data["Y displacement (nm)"] = data['Y displacement (pixels)']*pixel_size

    ## Step 3; Create time step
    data["Time (ms)"] = data['index']*time_step

    # Recalculation of center using distance forumla -- Jerry
    # Radius Calculation from distance formula
    data['Radius (nm)'] = np.power(((data["X displacement (nm)"])
                                    ** 2 + (data["Y displacement (nm)"])**2), 0.5)

    # Z score calculation
    import scipy.stats as stats  # added to calculate z-score for Radius filtering
    data['z-score Rad'] = stats.zscore(data["Radius (nm)"])

    # Angle Calculation

    # Radian to degree conversion factor
    r2d = 180/np.pi

    # Take Arc Tan function of x and y coord to get radius. Arctan 2 makes Quad 3 and 4 negative.
    data['Angle'] = -np.arctan2(data['Y displacement (nm)'],
                                data['X displacement (nm)'])*r2d

    # Make all negative Theta values positive equivalents
    data.loc[data.Angle < 0, ['Angle']] += 360

    
    return data

def downsample(data,pars = pars, **kwargs):

    #estabilsh processing type
    processing = kwargs.get('processing',pars.processing)

    if processing == "none":
        df = data
        return df
    
    if processing == "moving average":

        bin_size = kwargs.get('bin_size',pars.downsampling_bin_size)

         # Simple Moving Average or "filter" dataframe:
        ma = pd.DataFrame(data.iloc[:, 0], columns=['index'])
        window = bin_size

        # for each column after the first (index) apply moving average filter
        for col in data.columns[1:]:
            ma[col] = data[col].rolling(window=window).mean()

        # Remove NaN's
        # In moving avg, all indices less than bin_size will be NaN 
        ma = ma.apply(pd.to_numeric, errors='coerce') 
        ma = ma.dropna()

        # Reset index
        ma = ma.reset_index(drop=True)
        
        df = ma
        return df
    
    if processing == "downsample":
        # Create copy of data dataframe
        da = data.copy()

        # define downsampling average function
        # This function taken from https://stackoverflow.com/questions/10847660/subsampling-averaging-over-a-numpy-array
        # allows us to downsample by averages over a set number
        # (change 'n' to the number of values you want to average over)

        def average_column(df, col_num, n):
            arr = df.iloc[:, col_num].values 
            end = n * int(len(arr)/n) 
            return np.mean(arr[:end].reshape(-1, n), 1)

        # Iterate over columns of da except for the first column (index column)
        col_names = list(da.columns)
        averaged_cols = []
        for col_name in col_names:
            averaged_col = average_column(da, da.columns.get_loc(col_name), bin_size)
            averaged_cols.append(averaged_col)

        # Combine the averaged columns into a new dataframe
        dsa = pd.DataFrame(averaged_cols).T
        dsa.columns = da.columns

        import math
        start_frame = math.floor(start_frame/bin_size)
        end_frame = math.floor(end_frame/bin_size)

        df = dsa
        return df, start_frame, end_frame
    print(f"Data has been processed:{processing}")

def plot_2D_graph(df,fig = None, pars=pars, **kwargs):
    ''''''

    # load kwargs
    start_frame = kwargs.get('start_frame',pars.start_frame)
    end_frame = kwargs.get('end_frame',pars.end_frame)
    unit = kwargs.get('unit',pars.unit)
    cmap = kwargs.get('cmap',pars.cmap)
    marker_size = kwargs.get('marker_size',pars.marker_size)
    display_center = kwargs.get('display_center',pars.display_center)
    expected_radius = kwargs.get('expected_radius',pars.expected_radius)
    pixel_min = kwargs.get('pixel_min', pars.pixel_min)
    pixel_max = kwargs.get('pixel_max', pars.pixel_max)
    nm_min = kwargs.get('nm_min', pars.nm_min)
    nm_max = kwargs.get('nm_max', pars.nm_max)
    axis_increment_pixel = kwargs.get('axis_increment_pixel', pars.axis_increment_pixel)
    axis_increment_nm = kwargs.get('axis_increment_nm', pars.axis_increment_nm)
    title = kwargs.get('title',pars.title)

    #import
    import matplotlib.pyplot as plt

    # If figure and axes are not predefined inputs, make them
    if fig is None:
        fig, ax = plt.subplots(1,1,figsize=(7, 6))
    else:
        ax = fig.add_subplot(111)
    
    # Data assignment
    # Here the code determines the units of the graph, only for cartesian graphs
    if unit == "pixel":
        x = df["X displacement (pixels)"]
        y = df["Y displacement (pixels)"]
    if unit == "nm":
        x = df["X displacement (nm)"]
        y = df["Y displacement (nm)"]
    z = df["Time (ms)"]
   
    
    
        

    # Set up for color bar
    z_axis_label = "Frames" 

    # A color bar associated with time needs two things c and cmap
    #these arguments go into ax.scatter as args

    # c (A scalar or sequence of n numbers to be mapped to colors using cmap and norm.)
    c = df["index"]

    #Make a ticks vector that spans the total number of frames
    # There is a bug because linspace doesn't understand what -1 is but the sequence does
    if end_frame == -1:   # negative 1
        last_frame = df["index"].iat[-1] #in the index column, give me the last valid value --> this is the max Frames
    else:
        last_frame = end_frame

    frame_step=int((last_frame-start_frame)/5)

    tix_1=np.arange(start_frame,last_frame,frame_step)

    #scatter plot with a color vector
    p = ax.scatter(x, y, c=c, cmap = cmap, alpha=0.7, s=marker_size)
    #add a vertical side bar that defines the color
    plt.colorbar(p, label=z_axis_label, shrink=.82, ticks=tix_1)
    # plt.colorbar(p, label=z_axis_label, shrink=.82)


    plt.axis('square')
    plt.xticks(rotation=45)
    circle2 = plt.Circle((0, 0), expected_radius, color='m', fill=False)
    
    ax.add_patch(circle2)

    # display center
    if display_center == 1:
        # in a centered graph, the center is actually(0,0)
        center1 = [0, 0]
        # plots center point as magenta X
        ax.scatter(0, 0, color='Magenta', marker="X", s=150)
        plt.text(x=center1[0] + 0.02,
                    y=center1[1] + 0.02, s='CENTER')

    # set graph limit conditions depending on unit specified
    if unit == "pixel":
        ax.set_xlim(pixel_min, pixel_max)
        ax.set_ylim(pixel_min, pixel_max)
        # Set the x and y tick increments
        ax.set_xticks(np.arange(pixel_min, pixel_max, axis_increment_pixel))
        ax.set_yticks(np.arange(pixel_min, pixel_max, axis_increment_pixel))
        x_axis_label = "x (px)"
        y_axis_label = "y (px)"
    if unit == "nm":

        # Set x and y limits
        ax.set_xlim(nm_min, nm_max)
        ax.set_ylim(nm_min, nm_max)
        
        # Set the x and y tick increments
        ax.set_xticks(np.arange(nm_min, nm_max, axis_increment_nm))
        ax.set_yticks(np.arange(nm_min, nm_max, axis_increment_nm))
        
        # Set x and y labels
        x_axis_label = "x (nm)"
        y_axis_label = "y (nm)"
    
    # Jerry Adds a hover cursor
    # mplcursors.cursor(hover=True)
    # mplcursors.cursor(highlight=True)

    # Title axis labels and font configurations
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
    ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)

    
    # plot title and font configurations

    # plt.title(pk, fontweight='bold', fontsize=16)
