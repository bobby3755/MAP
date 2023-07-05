"""
This module contains the supporting functions necessary to execute DORA. These supporting functions are called within the GUI (MAP.x.y.py). DORA parameters are read in a DORA2_settings XML.
"""

import sma_lib.MAP_Parameters as params
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats  # added to calculate z-score for Radius filtering

xmlname = "DORA2_settings"

#read in the settings in the .xml file using hazen's Parameter Class
pars = params.Parameters(xmlname+'.xml') #par is an object of type Parameters, defined in sa_library
#to access parameters, use par.parameter name. eg par.start_frame
#note these values can be manually changed: par.frameset = 200 replaces whatever was there.
# print(pars.end_frame)

def load_csv(selected_csv, dir_path, pars=pars, **kwargs):
    """Loads a CSV file and organizes the data frame, by trimming frames to start and ending, naming columns, and adding a time step column.

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

    Raises:
        FileNotFoundError: If the specified CSV file is not found in the given directory path.


    """

    # Extract values from XML
    start_frame = kwargs.get('start_frame', pars.start_frame)
    end_frame = kwargs.get('end_frame', pars.end_frame)
    time_step = kwargs.get('time_step', pars.time_step)


    # Read CSV file into Pandas DataFrame
    csv_path = os.path.join(dir_path, selected_csv)
    pre_data = pd.read_csv(csv_path)

    # Add an index column as the first column
    pre_data.insert(0, 'index', range(len(pre_data)))

    # Rename the first seven columns as 'index', 'X position', 'Y position', 'Intensity', 'Signal','Noise','SNR'
    pre_data.columns.values[:7] = ['index', 'X position', 'Y position', 'Intensity','Signal','Noise','SNR']

    # Create a 'Time (ms)' column based on the index and time_step
    pre_data.insert(1, 'Time (ms)', pre_data['index'] * time_step)

    # Section data from frame start to frame end
    pre_data = pre_data.iloc[start_frame:end_frame]

    print(f"[INFO] CSV Loaded: {selected_csv}")

    # Display the updated DataFrame
    print(pre_data.head(5))

    return pre_data


def remove_invalid_readings(data):
    """
    Removes invalid readings from the given DataFrame by adding boolean mask column to input DataFrame.

    Args:
        data (pandas.DataFrame): The DataFrame containing the readings.

    Notes:
        Invalid readings are identified as those with X position equal to 0.

    """
    # Create a boolean mask where True indicates invalid readings (X position == 0)
    invalid_mask = data['X position'] == 0

    num_invalid_readings = invalid_mask.sum()

    data['err_invalid_reading'] = invalid_mask

    print(f"[INFO] Number of Invalid Readings Removed: {num_invalid_readings}")

    
def remove_nopass_intensity_filter(data, min_intensity=None, max_intensity=None):
    """
    Removes readings that do not pass the intensity filter from the given DataFrame by adding boolean mask column to input DataFrame.

    Args:
        data (pandas.DataFrame): The DataFrame containing the readings.
        min_intensity (float, optional): The minimum intensity threshold. If not specified, no lower limit is applied.
        max_intensity (float, optional): The maximum intensity threshold. If not specified, no upper limit is applied.

    Notes:
        Readings that fall below the minimum intensity or exceed the maximum intensity are considered invalid.

    """
    # Create error columns
    data["err_intensity_filter_low"] = data["Intensity"] < min_intensity
    data["err_intensity_filter_high"] = data["Intensity"] > max_intensity
     
    clean_data = remove_all_errors(data)

    # Calculate the number of errors and notify the user
    num_err_intensity_filter_low = data['err_intensity_filter_low'].sum()
    num_err_intensity_filter_high = data['err_intensity_filter_high'].sum()

    print(f"[INFO] Number of readings below the minimum intensity removed: {num_err_intensity_filter_low}")
    print(f"[INFO] Number of readings above the maximum intensity removed: {num_err_intensity_filter_high}")


def remove_all_errors(input_data):
    """
    Removes rows from the input DataFrame that contain any errors. This enables the user to work with the clean data.

    Args:
        input_data (pandas.DataFrame): The DataFrame containing the data with error columns.

    Returns:
        pandas.DataFrame: The cleaned DataFrame with rows containing errors removed.

    Notes:
        This function filters out rows based on the presence of any 'True' values in columns starting with 'err_'.

    """
    # Make a copy of the input data to avoid modifying the original DataFrame
    data = input_data.copy()

    # Find all columns that start with 'err_'
    error_columns = data.columns[data.columns.str.startswith('err_')]

    # Check if any of the error columns have 'True' values
    error_mask = data[error_columns].any(axis=1)

    # Return the cleaned DataFrame by excluding rows with errors
    return data[~error_mask]


def find_center(input_data, pars=pars, **kwargs):
    """
    Finds the center of the data using a specified centering strategy.

    Args:
        input_data (pandas.DataFrame): The input DataFrame containing the data.
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - centering_strategy (str): The centering strategy to use (default: pars.centering_strategy).
            - bin_num (int): The number of bins for the low-resolution histogram (default: pars.center_hist_max_bins).
            - pixel_size (float): The pixel size (default: pars.pixel_size).

    Returns:
        tuple: A tuple containing the center coordinates and the radius estimate (center, radius_estimate).

    Notes:
        - This function supports two centering strategies: "center_hist_max" and "center_circular_trajectory".
        - The centering strategy "center_hist_max" finds the center based on the maximum value in a low-resolution histogram.
        - The centering strategy "center_circular_trajectory" finds the center using circular trajectory analysis, originally developed by Claire Rickets.
        - The radius_estimate is only a useful paramter in center_circular_trajectory, which generates the projected radius.

    """

    # Extract values from XML
    centering_strategy = kwargs.get('centering_strategy', pars.centering_strategy)
    pixel_size = kwargs.get('pixel_size', pars.pixel_size)

    # Make a copy of the input data to avoid modifying the original DataFrame
    data = input_data.copy()

    # Remove rows with errors
    data = remove_all_errors(data)

    # Extract x and y values
    x = data["X position"]
    y = data["Y position"]

    # Centering strategy: "center_hist_max"
    if centering_strategy == "center_hist_max":
        # Extract values from XML
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
        print(f"[INFO] From {centering_strategy}, the center is {center} with a radius of {round(radius_estimate * pixel_size, 4)} nm")

    return center, radius_estimate


def generate_centered_data(input_data, center):
    """
    Generates centered data by subtracting the center coordinates.

    Args:
        input_data (pandas.DataFrame): The input DataFrame containing the data.
        center (tuple): The center coordinates (x, y).

    Returns:
        tuple: The x and y displacement arrays after centering.

    """

    # Make a copy of the input data to avoid modifying the original DataFrame
    data = input_data.copy()

    # Remove rows with errors
    data = remove_all_errors(data)

    # Extract x and y values
    x = data["X position"]
    y = data["Y position"]

    # Subtract center coordinates from each position column to find displacement and store in new columns
    x_centered = x - center[0]
    y_centered = y - center[1]

    print(f"[INFO] Data has been centered around {center[0]}, {center[1]}")

    return x_centered, y_centered

    
def calculate_radius(data):
    """
    Calculates the radius based on x and y displacements.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        numpy.ndarray: The calculated radius values.

    """
    x = data["X displacement (pixel)"]
    y = data["Y displacement (pixel)"]
    rad = np.sqrt(x**2 + y**2)

    return rad

def calculate_SNR2(input_data):
    """
    Calculates the signal to noise from signal and noise columns.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        numpy.ndarray: The calculated radius values.

    """
    # Clean Data
    data = remove_all_errors(input_data)

    # Add your calculation
    snr = data["Signal"]/data["Noise"]

    # Notify User
    print("[INFO] SNR2 has been calculated")

    return snr

def calculate_rad_zscore(data):
    """
    Calculates the z-score for the radius values in the given data.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        numpy.ndarray: The z-score values for the radius.

    Notes:
        This function is intended to be used as part of a filtering analysis to identify and remove outlier radii. 
        By calculating the z-score for each radius value, you can compare it to the distribution of radii and determine 
        which values are significantly different from the mean. You can then apply a threshold to remove radii 
        that fall beyond a certain z-score, effectively filtering out outliers. (Full functionality has not been addded yet)
    """
    zscore_rad = stats.zscore(data["Radius (pixel)"].dropna())
    return zscore_rad


def calculate_angle(input_data, pars=pars, **kwargs):
    """
    Calculates the angle, differnetial angle, and continuous based on the displacement values in the given data.

    Args:
        input_data (pandas.DataFrame): The input DataFrame containing the data.
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - rad_filter_type_lower (str): The lower filter type for the radius (default: pars.rad_filter_type_lower).
            - rad_filter_type_upper (str): The upper filter type for the radius (default: pars.rad_filter_type_upper).
            - z_low (float): The lower threshold for z-score filtering (default: pars.z_low).
            - z_high (float): The upper threshold for z-score filtering (default: pars.z_high).
            - dist_low (float): The lower threshold for distance filtering (default: pars.dist_low).
            - dist_high (float): The upper threshold for distance filtering (default: pars.dist_high).

    Returns:
        tuple: A tuple containing the angle, delta angle, and continuous angle arrays.

    Notes:
        - The angle is calculated based on the displacement values (X displacement and Y displacement) in the input data. 
        - It uses the arctan2 function to calculate the angle.Negative angles are converted to their positive equivalents. 
        - The function also calculates the delta angle and the continuous angle by applying differential and cumulative summation, respectively.
        - The delta angle calculation assumes that there is not enough time for the probe to travel more than 180 degrees, prefering the shorter path around the circle.

        
    """
    data = remove_all_errors(input_data)

    rad_filter_type_lower = kwargs.get('rad_filter_type_lower', pars.rad_filter_type_lower)
    rad_filter_type_upper = kwargs.get('rad_filter_type_upper', pars.rad_filter_type_upper)
    z_low = kwargs.get('z_low', pars.z_low)
    z_high = kwargs.get('z_high', pars.z_high)
    dist_low = kwargs.get('dist_low', pars.dist_low)
    dist_high = kwargs.get('dist_high', pars.dist_high)

    r2d = 180 / np.pi

    angle = -np.arctan2(data['Y displacement (pixel)'], data['X displacement (pixel)']) * r2d
    angle[angle < 0] += 360

    # Marginal Angle calculation using the my_diff function
    def calculate_delta_angle(vec):

        vect = vec.diff()  # run a differential on all the angles

        vect[0] = 0  # set the first NaN to 0

        # assuming all increments are less than 180,
        # then make all changes bigger than 180, less than 180.

        # greater than 180 --> negative equivalent
        vect[vect >= (180)] -= 360

        # less than -180 --> positive equivalent
        vect[vect <= (-180)] += 360

        return vect

    delta_angle = calculate_delta_angle(angle)
    continuous_angle = delta_angle.cumsum()

    print("[INFO] Angle calculations have been calculated")

    return angle, delta_angle, continuous_angle


def downsample(data, pars=pars, **kwargs):
    """
    Performs downsampling on the data based on the specified processing type.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - processing (str): The type of processing to apply (default: pars.processing).
            - bin_size (int): The size of the bin for moving average or downsampling (default: pars.downsampling_bin_size).

    Returns:
        pandas.DataFrame: The downsampled DataFrame based on the specified processing type.

    Notes:
        - If the processing type is "none", the original data is returned.
        - If the processing type is "moving average", a moving average filter is applied to each column after the first.
        - If the processing type is "downsample", the data is downsampled by averaging values within each bin.

    """

    # Extract parameters from XML
    processing = kwargs.get('processing', pars.processing)

    if processing == "none":
        df = data
        print(f"[INFO] Data has been processed: {processing}")
        return df

    if processing == "moving average":

         # Extract bin_size from XML 
        bin_size = kwargs.get('pars.downsampling_bin_size', pars.downsampling_bin_size)

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
        
        # Extract bin_size from XML 
        bin_size = kwargs.get('pars.downsampling_bin_size', pars.downsampling_bin_size)
        
        # Create a copy of the data dataframe
        da = data.copy()

        def average_column(df, col_num, n):

            # Extract the values from the specified column
            column_values = df.iloc[:, col_num].values
            
            # Calculate the index at which downsampling should end
            end_index = n * (len(column_values) // n)
            
            # Slice the array and reshape it into a 2D array with 'n' columns
            sliced_array = column_values[:end_index].reshape(-1, n)
            
            # Calculate the mean along each row of the reshaped array
            averaged_values = np.mean(sliced_array, axis=1)
            
            return averaged_values

        # Iterate over columns of da (except for the first index column)
        col_names = list(da.columns)
        averaged_cols = []
        for col_name in col_names:
            averaged_col = average_column(da, da.columns.get_loc(col_name), bin_size)
            averaged_cols.append(averaged_col)

        # Combine the averaged columns into a new dataframe
        dsa = pd.DataFrame(averaged_cols).T
        dsa.columns = da.columns

        print(f"[INFO] Data has been processed: {processing}")

        return dsa

    
def plot_2D_graph(df, fig=None, column_headers=None, pars=pars, **kwargs):
    """
    Plots a 2D graph based on the provided DataFrame. This is a graph of the X,Y,Time values.

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
            - num_ticks (int): The number of ticks on the color bar (default: pars.num_ticks).
            - axis_increment_pixel (float): The increment value for the pixel axis ticks (default: pars.axis_increment_pixel).
            - axis_increment_nm (float): The increment value for the nm axis ticks (default: pars.axis_increment_nm).
            - title (str): The title of the graph (default: pars.title).

    Returns:
        matplotlib.figure.Figure: The generated figure.

    """

    # Extract values from XML
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


def plot_angular_continuous(input_data, fig=None, pars=pars, **kwargs):
    """
    Plots the continuous angle vs. time based on the provided input data.

    Args:
        input_data (pandas.DataFrame): The input DataFrame containing the data.
        fig (matplotlib.figure.Figure): Optional predefined figure (default: None).
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - angle_vs_time_style (str): The style of the plot ('scatter' or 'line') (default: pars.angle_vs_time_style).
            - angle_vs_time_color (str): The color of the plot (default: pars.angle_vs_time_color).
            - angle_vs_time_xlabel (str): The x-axis label ('Frames' or 'Time (ms)') (default: pars.angle_vs_time_xlabel).
            - filter_nopass (bool): Whether to remove readings with errors (default: pars.filter_nopass).
            - annotate_nopass (bool): Whether to annotate removed readings with errors (default: pars.annotate_nopass).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Extract values from XML
    angle_vs_time_style = kwargs.get('angle_vs_time_style', pars.angle_vs_time_style)
    angle_vs_time_color = kwargs.get('angle_vs_time_color', pars.angle_vs_time_color)
    angle_vs_time_xlabel = kwargs.get('angle_vs_time_xlabel', pars.angle_vs_time_xlabel)

    # If figure is not predefined, create a new figure
    if fig is None:
        fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(5.5,5)s
    else:
        # fig.set_size_inches(5,5)
        fig.clf()
        ax = fig.add_subplot(111)

    # Remove readings with erros
    df = remove_all_errors(input_data)

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
    

def plot_intensity_time(input_data, fig=None, pars=pars, **kwargs):
    """
    Plots the intensity vs. time based on the provided input data.

    Args:
        input_data (pandas.DataFrame): The input DataFrame containing the data.
        fig (matplotlib.figure.Figure): Optional predefined figure (default: None).
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - intensity_vs_time_style (str): The style of the plot ('scatter' or 'line') (default: pars.intensity_vs_time_style).
            - intensity_vs_time_color (str): The color of the plot (default: pars.intensity_vs_time_color).
            - intensity_vs_time_xlabel (str): The x-axis label ('Frames' or 'Time (ms)') (default: pars.intensity_vs_time_xlabel).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    
    # Extract parameters from XML
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

    # Remove readings with erros
    df = remove_all_errors(input_data)

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

    # Set the minimum y-value
    # ax.set_ylim(bottom=0)

    # Set Title and y axis label
    title = "Intensity vs Frames"
    y_axis_label = 'Intensity (a.u.)'

    # Set title, axis labels, and font configurations
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
    ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)


def plot_snr_time(input_data, fig=None, pars=pars, **kwargs):
    """
    Plots the SNR vs. time based on the provided input data.

    Args:
        input_data (pandas.DataFrame): The input DataFrame containing the data.
        fig (matplotlib.figure.Figure): Optional predefined figure (default: None).
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - snr_vs_time_style (str): The style of the plot ('scatter' or 'line') (default: pars.snr_vs_time_style).
            - snr_vs_time_color (str): The color of the plot (default: pars.snr_vs_time_color).
            - snr_vs_time_xlabel (str): The x-axis label ('Frames' or 'Time (ms)') (default: pars.snr_vs_time_xlabel).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    
    # Extract parameters from XML
    snr_vs_time_style = kwargs.get('snr_vs_time_style', pars.snr_vs_time_style)
    snr_vs_time_color = kwargs.get('snr_vs_time_color', pars.snr_vs_time_color)
    snr_vs_time_xlabel = kwargs.get('snr_vs_time_xlabel', pars.snr_vs_time_xlabel)

     # If figure is not predefined, create a new figure
    if fig is None:
        fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(5.5,5)s
    else:
        # fig.set_size_inches(5,5)
        fig.clf()
        ax = fig.add_subplot(111)

    # Remove readings with erros
    df = remove_all_errors(input_data)

    # define x axlis label
    if snr_vs_time_xlabel == "Frames":
        frames = df["index"]
        x_axis_label = "Frames"
    elif snr_vs_time_xlabel == "Time (ms)":
        frames = df["Times (ms)"]
        x_axis_label = "Time (ms)"
    else:
        ValueError("[ERROR] entered snr_vs_time_xlabel is one of the executable options.")

    # choose scatter plot or line plot
    if snr_vs_time_style == 'scatter':
        snr_graph = ax.scatter(frames, df["SNR"], color=snr_vs_time_color, s=5)
    elif snr_vs_time_style == 'line':
        snr_graph = ax.plot(frames, df["SNR"],  color=snr_vs_time_color)

    # Set the minimum y-value
    # ax.set_ylim(bottom=0)

    # Set Title and y axis label
    title = "snr vs Frames"
    y_axis_label = 'snr (a.u.)'

    # Set title, axis labels, and font configurations
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
    ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)

def plot_snr2_time(input_data, fig=None, pars=pars, **kwargs):
    """
    Plots the snr2 vs. time based on the provided input data.

    Args:
        input_data (pandas.DataFrame): The input DataFrame containing the data.
        fig (matplotlib.figure.Figure): Optional predefined figure (default: None).
        pars (object): Optional object containing default parameters (default: pars, as defined at the top of the script).
        **kwargs: Additional keyword arguments.
            - snr2_vs_time_style (str): The style of the plot ('scatter' or 'line') (default: pars.snr2_vs_time_style).
            - snr2_vs_time_color (str): The color of the plot (default: pars.snr2_vs_time_color).
            - snr2_vs_time_xlabel (str): The x-axis label ('Frames' or 'Time (ms)') (default: pars.snr2_vs_time_xlabel).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    
    # Extract parameters from XML
    snr2_vs_time_style = kwargs.get('snr2_vs_time_style', pars.snr2_vs_time_style)
    snr2_vs_time_color = kwargs.get('snr2_vs_time_color', pars.snr2_vs_time_color)
    snr2_vs_time_xlabel = kwargs.get('snr2_vs_time_xlabel', pars.snr2_vs_time_xlabel)

     # If figure is not predefined, create a new figure
    if fig is None:
        fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(5.5,5)s
    else:
        # fig.set_size_inches(5,5)
        fig.clf()
        ax = fig.add_subplot(111)

    # Remove readings with erros
    df = remove_all_errors(input_data)

    # define x axlis label
    if snr2_vs_time_xlabel == "Frames":
        frames = df["index"]
        x_axis_label = "Frames"
    elif snr2_vs_time_xlabel == "Time (ms)":
        frames = df["Times (ms)"]
        x_axis_label = "Time (ms)"
    else:
        ValueError("[ERROR] entered snr2_vs_time_xlabel is one of the executable options.")

    # choose scatter plot or line plot
    if snr2_vs_time_style == 'scatter':
        avt_graph = ax.scatter(frames, df["SNR2"], color=snr2_vs_time_color, s=5)
    elif snr2_vs_time_style == 'line':
        avt_graph = ax.plot(frames, df["SNR2"],  color=snr2_vs_time_color)

    # Set the minimum y-value
    # ax.set_ylim(bottom=0)

    # Set Title and y axis label
    title = "SNR vs Frames"
    y_axis_label = 'snr (a.u.)'

    # Set title, axis labels, and font configurations
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel(x_axis_label, fontweight='bold', fontsize=14)
    ax.set_ylabel(y_axis_label, fontweight='bold', fontsize=14)
