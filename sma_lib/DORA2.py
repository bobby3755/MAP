import sma_lib.MAP_Parameters as params
import sma_lib.AngleCalc as AngleCalc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import mplcursors

xmlname = "DORA2_settings"

#read in the settings in the .xml file using hazen's Parameter Class
pars = params.Parameters(xmlname+'.xml') #par is an object of type Parameters, defined in sa_library
#to access parameters, use par.parameter name. eg par.start_frame
#note these values can be manually changed: par.frameset = 200 replaces whatever was there.
# print(pars.end_frame)

def load_csv(selected_csv, dir_path, pars=pars, **kwargs):
    """Loads a CSV file and organizes the data frame by trimming frames and adding a time step column.

    Args:
        selected_csv (str): The name of the selected CSV file.
        dir_path (str): The directory path where the CSV file is located.
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments for customizing frame trimming:
            - start_frame (int): The starting frame index (default: pars.start_frame).
            - end_frame (int): The ending frame index (default: pars.end_frame).
            - time_step (float): The time step value (default: pars.time_step).

    Returns:
        pandas.DataFrame: The trimmed DataFrame containing the data.

    """

    # Load kwargs
    start_frame = kwargs.get('start_frame', pars.start_frame)
    end_frame = kwargs.get('end_frame', pars.end_frame)
    time_step = kwargs.get('time_step', pars.time_step)


    # Read CSV file into Pandas DataFrame
    csv_path = os.path.join(dir_path, selected_csv)
    pre_data = pd.read_csv(csv_path)

    # Add an index column as the first column
    pre_data.insert(0, 'index', range(len(pre_data)))

    # Rename the first four columns as 'index', 'X position', 'Y position', and 'Intensity'
    pre_data.columns.values[:4] = ['index', 'X position', 'Y position', 'Intensity']

    # Create a 'Time (ms)' column based on the index and time_step
    pre_data.insert(1, 'Time (ms)', pre_data['index'] * time_step)

    # Section data from frame start to frame end
    pre_data = pre_data.iloc[start_frame:end_frame]

    print(f"[INFO] CSV Loaded: {selected_csv}")

    # Display the updated DataFrame
    print(pre_data.head(5))

    return pre_data


def remove_invalid_readings(pre_data):
    """Remove invalid readings from the data and export filtered data.

    Args:
        pre_data (pandas.DataFrame): The input DataFrame containing the readings.

    Returns:
        tuple: A tuple containing:
            - data (pandas.DataFrame): The filtered data without invalid readings.
            - ind_invalid_reading (pandas.Series): A boolean series indicating the invalid readings.
            - data_back (pandas.DataFrame): A DataFrame containing the filtered invalid readings.

    """
    # Create a boolean array where True indicates invalid readings (X position == 0)
    ind_invalid_reading = pre_data['X position'] == 0

    # Separate valid data and invalid readings
    data = pre_data[~ind_invalid_reading].copy()
    data_back = pre_data[ind_invalid_reading].copy()

    # Set X and Y positions to NaN for invalid readings
    data_back.loc[ind_invalid_reading, ['X position', 'Y position']] = np.nan
    data_back['Excluded Type'] = 'Invalid Reading'

    num_invalid_readings = ind_invalid_reading.sum()

    print(f"[INFO] Number of Invalid Readings Removed: {num_invalid_readings}")

    return data, ind_invalid_reading, data_back


def find_center(data, pars=pars, **kwargs):
    """Finds the center of the data using a specified centering strategy.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
         **kwargs: Additional keyword arguments.
            - centering_strategy (str): The centering strategy to use (default: pars.centering_strategy).
            - bin_num (int): The number of bins for the low-resolution histogram (default: pars.center_hist_max_bins).
            
    Returns:
        tuple: A tuple containing the center coordinates and the radius estimate (center, radius_estimate).

    """

    # Load kwargs
    centering_strategy = kwargs.get('centering_strategy', pars.centering_strategy)
    pixel_size = kwargs.get('pixel_size',pars.pixel_size)
    
    # Extract x and y values
    x = data["X position"]
    y = data["Y position"]

    # Centering strategy: "center_hist_max"
    if centering_strategy == "center_hist_max":

        # Load kwargs
        bin_num = kwargs.get('bin_num', pars.center_hist_max_bins)

        # Calculate a low-resolution histogram and find the maximum value
        H, xedges, yedges = np.histogram2d(x, y, bins=bin_num)

        # Find the x and y indices of the maximum value in the histogram
        indMax = np.unravel_index(H.argmax(), H.shape)

        # Set the max x and y indices as the new center coordinates
        center = [xedges[indMax[0]], yedges[indMax[1]]]
        radius_estimate = None
        print(f"[INFO] From {centering_strategy}, the center is {center}")

    # Centering strategy: "center_circular_trajectory"
    if centering_strategy == "center_circular_trajectory":

        total_guesses = 10000
        n = 2
        num_points = int(total_guesses ** (1 / n))

        # Generate uniform guesses for y position
        guess_y = np.linspace(y.max(), y.min(), num_points)

        # Generate guesses for x position
        guess_x = np.linspace(x.max(), x.min(), num_points)

        # Generate center guesses by permuting x and y guesses
        center_x, center_y = np.meshgrid(guess_x, guess_y)
        center_guesses = np.column_stack((center_x.ravel(), center_y.ravel()))

        # Calculate distances between each point in the data and all center guesses
        distances = np.sqrt(
            (np.array(data['X position'])[:, np.newaxis] - center_guesses[:, 0]) ** 2 +
            (np.array(data['Y position'])[:, np.newaxis] - center_guesses[:, 1]) ** 2
        )

        # Calculate average distances (radii) for each center guess
        ave_distance = np.mean(distances, axis=0)

        # Calculate standard deviation of distances for each center guess
        std = np.std(distances, axis=0)

        # Find the index of the center guess with the lowest std
        target_row = np.argmin(std)

        # Retrieve the center coordinates and radius for the best guess
        center_x = center_guesses[target_row, 0]
        center_y = center_guesses[target_row, 1]
        radius_estimate = ave_distance[target_row]
        center = (center_x, center_y)
        print(f"[INFO] From {centering_strategy}, the center is {center} with a radius of {round(radius_estimate * pixel_size,4)} nm")

    return center, radius_estimate


def generate_centered_data(data, center):
    """Generates centered data by subtracting the center coordinates.

    Args:
        x (pandas.Series or array-like): The x-coordinate data.
        y (pandas.Series or array-like): The y-coordinate data.
        center (tuple): The center coordinates (x, y).

    Returns:
        tuple: The x and y displacement arrays after centering.

    """

    # Extract x and y values
    x = data["X position"]
    y = data["Y position"]

    # Subtract center coordinates from each position column to find displacement and store in new columns
    data["X displacement (pixel)"] = x - center[0]
    data["Y displacement (pixel)"] = y - center[1]

    return data


def calculate_angle(data, pars=pars, **kwargs):
    """Calculates the angle based on the given data.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - processing (str): The type of processing to apply (default: pars.processing).
            - bin_size (int): The size of the bin for moving average or downsampling (default: pars.downsampling_bin_size).
            - start_frame (int): The starting frame index for downsampling (default: pars.start_frame).
            - end_frame (int): The ending frame index for downsampling (default: pars.end_frame).

    Returns:
        pandas.DataFrame: The DataFrame with calculated angle values.

    """

    # Recalculation of center using distance formula -- Jerry
    data['Radius (pixel)'] = np.sqrt(data["X displacement (pixel)"]**2 + data["Y displacement (pixel)"]**2)
    
    # Z score calculation
    import scipy.stats as stats  # Importing scipy.stats for calculating z-score
    data['z-score Rad'] = stats.zscore(data["Radius (pixel)"])


    # Angle Calculation

    # Radian to degree conversion factor
    r2d = 180 / np.pi

    # Calculate the angle using arctan2 function
    data['Angle'] = -np.arctan2(data['Y displacement (pixel)'], data['X displacement (pixel)']) * r2d

    # Make all negative Theta values positive equivalents
    data.loc[data.Angle < 0, 'Angle'] += 360

    ##### Start to calculate angular continuous

    #Calculate
    data, data_filtered_pass, data_filtered_nopass, data_filtered_lower_bound_nopass, data_filtered_upper_bound_nopass = AngleCalc.conti_angle_calc(data)
    print("[INFO] Updated angle calculations have been calculated")

    return data


def downsample(data, pars=pars, **kwargs):
    """Performs downsampling on the data based on the specified processing type.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - processing (str): The type of processing to apply (default: pars.processing).
            - bin_size (int): The size of the bin for moving average or downsampling (default: pars.downsampling_bin_size).
            - start_frame (int): The starting frame index for downsampling (default: pars.start_frame).
            - end_frame (int): The ending frame index for downsampling (default: pars.end_frame).

    Returns:
        pandas.DataFrame: The downsampled DataFrame based on the specified processing type.

    """

    # Establish processing type from kwargs
    processing = kwargs.get('processing', pars.processing)

    if processing == "none":
        df = data
        print(f"[INFO] Data has been processed: {processing}")
        return df

    if processing == "moving average":
        bin_size = kwargs.get('bin_size', pars.downsampling_bin_size)

        # Simple Moving Average or "filter" dataframe:
        ma = pd.DataFrame(data.iloc[:, 0], columns=['index'])
        window = bin_size

        # Apply moving average filter to each column after the first (index)
        for col in data.columns[1:]:
            ma[col] = data[col].rolling(window=window).mean()

        # Remove NaN values (indices less than bin_size will be NaN)
        ma = ma.apply(pd.to_numeric, errors='coerce')
        ma = ma.dropna()

        # Reset index
        ma = ma.reset_index(drop=True)

        print(f"Data has been processed: {processing}")

        return ma

    if processing == "downsample":
        # Create a copy of the data dataframe
        da = data.copy()

        # Define downsampling average function
        def average_column(df, col_num, n):
            arr = df.iloc[:, col_num].values
            end = n * int(len(arr) / n)
            return np.mean(arr[:end].reshape(-1, n), 1)

        # Iterate over columns of da (except for the first index column)
        col_names = list(da.columns)
        averaged_cols = []
        for col_name in col_names:
            averaged_col = average_column(da, da.columns.get_loc(col_name), bin_size)
            averaged_cols.append(averaged_col)

        # Combine the averaged columns into a new dataframe
        dsa = pd.DataFrame(averaged_cols).T
        dsa.columns = da.columns

        # Adjust start_frame and end_frame
        start_frame = kwargs.get('start_frame', pars.start_frame)
        end_frame = kwargs.get('end_frame', pars.end_frame)
        start_frame = math.floor(start_frame / bin_size)
        end_frame = math.floor(end_frame / bin_size)

        print(f"Data has been processed: {processing}")

        return dsa, start_frame, end_frame

    
def plot_2D_graph(df, fig=None, column_headers=None, pars=pars, **kwargs):
    """Plots a 2D graph based on the provided DataFrame.

    Args:
        df (pandas.DataFrame or numpy.ndarray): The input DataFrame or array containing the data.
        fig (matplotlib.figure.Figure): Optional predefined figure (default: None).
        column_headers (list): Optional list of column headers for the array data (default: None).
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - unit (str): The unit of measurement for the displacement values ('pixel' or 'nm') (default: pars.unit).
            - cmap (str): The colormap to use for the scatter plot (default: pars.cmap).
            - marker_size (float): The size of the markers in the scatter plot (default: pars.marker_size).
            - display_center (int): Whether to display the center point on the graph (1 for yes, 0 for no) (default: pars.display_center).
            - expected_radius (float): The expected radius for the circular marker (default: pars.expected_radius).
            - pixel_min (float): The minimum value for the pixel axis (default: pars.pixel_min).
            - pixel_max (float): The maximum value for the pixel axis (default: pars.pixel_max).
            - nm_min (float): The minimum value for the nm axis (default: pars.nm_min).
            - nm_max (float): The maximum value for the nm axis (default: pars.nm_max).
            - axis_increment_pixel (float): The increment value for the pixel axis ticks (default: pars.axis_increment_pixel).
            - axis_increment_nm (float): The increment value for the nm axis ticks (default: pars.axis_increment_nm).
            - title (str): The title of the graph (default: pars.title).

    Returns:
        matplotlib.figure.Figure: The generated figure.

    """

    # Load kwargs
    unit = kwargs.get('unit', pars.unit)
    cmap = kwargs.get('cmap', pars.cmap)
    marker_size = kwargs.get('marker_size', pars.marker_size)
    display_center = kwargs.get('display_center', pars.display_center)
    expected_radius = kwargs.get('expected_radius', pars.expected_radius)
    pixel_size = kwargs.get('pixel_size',pars.pixel_size)
    pixel_min = kwargs.get('pixel_min', pars.pixel_min)
    pixel_max = kwargs.get('pixel_max', pars.pixel_max)
    nm_min = kwargs.get('nm_min', pars.nm_min)
    nm_max = kwargs.get('nm_max', pars.nm_max)
    num_ticks = kwargs.get('num_ticks', pars.num_ticks)
    axis_increment_pixel = kwargs.get('axis_increment_pixel', pars.axis_increment_pixel)
    axis_increment_nm = kwargs.get('axis_increment_nm', pars.axis_increment_nm)
    title = kwargs.get('title', pars.title)


    # If figure is not predefined, create a new figure
    if fig is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(5.5,5)
    else:
        fig.clf()
        fig.set_size_inches(5.5,5)
        ax = fig.add_subplot(111)

    if isinstance(df, np.ndarray):
        # Convert array into DataFrame with columns
        df = pd.DataFrame(df, columns=column_headers)

    # Data assignment
    if unit == "pixel":
        x = df["X displacement (pixel)"]
        y = df["Y displacement (pixel)"]
    elif unit == "nm":
        x = df["X displacement (pixel)"] * pixel_size
        y = df["Y displacement (pixel)"] * pixel_size
    z = df["index"]

    # Scatter plot with color vector
    sc = ax.scatter(x, y, c=z, cmap=cmap, alpha=0.7, s=marker_size)

    # Set up color bar
    cbar = plt.colorbar(sc)
    z_axis_label = "Frames"
    
    # Calculate the tick locations
    num_ticks = num_ticks-1 # correction factor because the bottom one doesn't count internally
    min_val = np.min(z)
    max_val = np.max(z)
    tick_locs = np.arange(min_val, max_val + 1, (max_val - min_val) / num_ticks)

    # Set the tick locations on the color bar
    cbar.locator = ticker.FixedLocator(tick_locs)

    # Format the tick labels as integers
    cbar.formatter = ticker.FuncFormatter(lambda x, pos: f"{int(x)}")

    # Update the color bar with the new tick locations and labels
    cbar.update_ticks()

    


    plt.axis('square')
    plt.xticks(rotation=45)
    circle2 = plt.Circle((0, 0), expected_radius, color='m', fill=False)
    ax.add_patch(circle2)

    # Display center
    if display_center == 1:
        center1 = [0, 0]
        ax.scatter(0, 0, color='Magenta', marker="X", s=150)
        plt.text(x=center1[0] + 0.02, y=center1[1] + 0.02, s='CENTER')

    # Set graph limits and labels based on unit
    if unit == "pixel":
        ax.set_xlim(pixel_min, pixel_max)
        ax.set_ylim(pixel_min, pixel_max)
        ax.set_xticks(np.arange(pixel_min, pixel_max, axis_increment_pixel))
        ax.set_yticks(np.arange(pixel_min, pixel_max, axis_increment_pixel))
        x_axis_label = "x (px)"
        y_axis_label = "y (px)"
    elif unit == "nm":
        ax.set_xlim(nm_min, nm_max)
        ax.set_ylim(nm_min, nm_max)
        ax.set_xticks(np.arange(nm_min, nm_max, axis_increment_nm))
        ax.set_yticks(np.arange(nm_min, nm_max, axis_increment_nm))
        x_axis_label = "x (nm)"
        y_axis_label = "y (nm)"

    # Set title, axis labels, and font configurations
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
    ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)
    cbar.set_label(z_axis_label, fontweight='bold', fontsize=12)

    # # Add hover cursor
    # mplcursors.cursor(hover=True)
    # mplcursors.cursor(highlight=True)
def plot_angular_continuous(df, fig=None, pars=pars, **kwargs):
   
    # Load kwargs
    angle_vs_time_style = kwargs.get('angle_vs_time_style', pars.angle_vs_time_style)
    angle_vs_time_color = kwargs.get('angle_vs_time_color', pars.angle_vs_time_color)
    angle_vs_time_xlabel = kwargs.get('angle_vs_time_xlabel', pars.angle_vs_time_xlabel)
    filter_nopass = kwargs.get('filter_nopass', pars.filter_nopass)
    annotate_nopass = kwargs.get('annotate_nopass', pars.annotate_nopass) 

    # If figure is not predefined, create a new figure
    if fig is None:
        fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(5.5,5)s
    else:
        # fig.set_size_inches(5,5)
        fig.clf()
        ax = fig.add_subplot(111)

    # define x axlis label
    if angle_vs_time_xlabel == "Frames":
        times = df["index"]
        x_axis_label = "Frames"
    elif angle_vs_time_xlabel == "Time (ms)":
        times = df["Times (ms)"]
        x_axis_label = "Time (ms)"
    else:
        ValueError("[ERROR] entered angle_vs_time_xlabel is one of the executable options.")

    # choose scatter plot or line plot
    if angle_vs_time_style == 'scatter':
        avt_graph = ax.scatter(times, df["Continuous Angle"], color=angle_vs_time_color, s=5)
    elif angle_vs_time_style == 'line':
        avt_graph = ax.plot(times, df["Continuous Angle"],  color=angle_vs_time_color)

    # Set Title and y axis label
    title = "Continuous Angle vs time"
    y_axis_label = 'Angle Accumulation (degrees)'

    # Set title, axis labels, and font configurations
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
    ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)
    
   
    # Enable interactive cursor
    # cursor = mplcursors.cursor(hover=True)

    # Show the plot
    # plt.show()
    
    # Graph the newly calcuated Angular Continuous data, now filtered for good points only

def plot_intensity_time(df, fig=None, pars=pars, **kwargs):

    intensity_vs_time_style = kwargs.get('intensity_vs_time_style', pars.intensity_vs_time_style)
    intensity_vs_time_color = kwargs.get('intensity_vs_time_color', pars.intensity_vs_time_color)
    intensity_vs_time_xlabel = kwargs.get('intensity_vs_time_xlabel', pars.intensity_vs_time_xlabel)

     # If figure is not predefined, create a new figure
    if fig is None:
        fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(5.5,5)s
    else:
        # fig.set_size_inches(5,5)
        fig.clf()
        ax = fig.add_subplot(111)

    # define x axlis label
    if intensity_vs_time_xlabel == "Frames":
        frames = df["index"]
        x_axis_label = "Frames"
    elif intensity_vs_time_xlabel == "Time (ms)":
        frames = df["Times (ms)"]
        x_axis_label = "Time (ms)"
    else:
        ValueError("[ERROR] entered intensity_vs_time_xlabel is one of the executable options.")

    # choose scatter plot or line plot
    if intensity_vs_time_style == 'scatter':
        avt_graph = ax.scatter(frames, df["Intensity"], color=intensity_vs_time_color, s=5)
    elif intensity_vs_time_style == 'line':
        avt_graph = ax.plot(frames, df["Intensity"],  color=intensity_vs_time_color)

    # Set Title and y axis label
    title = "Intensity vs Frames"
    y_axis_label = 'Intensity (a.u.)'

    # Set title, axis labels, and font configurations
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
    ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)