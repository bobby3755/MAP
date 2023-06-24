"""
@author: Jerry Wu
Version: DORA 2.0
Last Update: 4.24.23

"""

import numpy as np
import pandas as pd
import sma_lib.MAP_Parameters as params
import os

# Determine the file path to the XML file
xml_path = os.path.join(os.path.dirname(__file__), '..', 'DORA2_settings.xml')

# Read in the settings from the XML file using Parameter class
pars = params.Parameters(xml_path)
# Access parameters using pars.parameter_name
    # For example:
    # print(pars.start_frame)



def conti_angle_calc(data, **kwargs):
    
    # set up block
    rad_filter_type_lower = kwargs.get('rad_filter_type_lower', pars.rad_filter_type_lower)
    rad_filter_type_upper = kwargs.get('rad_filter_type_upper', pars.rad_filter_type_upper)
    z_up = kwargs.get('z_up', pars.z_up)
    z_down = kwargs.get('z_down', pars.z_down)
    dist_low = kwargs.get('dist_low', pars.dist_low)
    dist_high = kwargs.get('dist_high', pars.dist_high)
    
    ######################################### ANGLE CALCULATION ###########################################

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

    # Assign down_sampled_df as data

    data_copy = data.copy(deep=True) #make a copy to remove questionable points later

    # _________________________________________[UNFLITERED DATA]__________________________________________________________

    #store Delta Angle
    data["Delta Angle"] = calculate_delta_angle(data["Angle"])

    # CONTINUOUS ANGLE: continuous sumation of the differentials
    # Run running summation fnction on differential angle
    data["Continuous Angle"] = data["Delta Angle"].cumsum()
    ################################ Filtered Data #############################################
    # Define variables for Radius and z-score radius columns
    my_rad = data_copy["Radius (pixel)"]  # Stores Radii into an array
    my_zscore = data_copy["z-score Rad"]  # Stores Z-scores into an array

    # Filter for good data: Upper BOUND
    if rad_filter_type_upper == "zscore":
        up_fil = my_zscore < z_up  
    else: # else use radius
        up_fil = my_rad < dist_high

    # Filter for good data: Lower BOUND
    if rad_filter_type_lower == "zscore":
        down_fil = z_down < my_zscore
    else: #else use radius
        down_fil = dist_low < my_rad
        # Filter for valid readings
    # Section data for Good data (within bounds)
    acceptable_ind = down_fil & up_fil

   
    # Put Time, X, Y, Radius, and Angle in a dataframe
    data_filtered_pass = data_copy.loc[:, ["Time (ms)", "X displacement (pixel)", "Y displacement (pixel)", "Radius (pixel)", "Angle"]]
    data_filtered_pass = data_filtered_pass[acceptable_ind]  # Keep only acceptable values

    # Section the eliminated data for both upper and lower
    # Put Time, X, Y, Radius, and Angle in a dataframe
    data_filtered_nopass = data_copy.loc[:,["Time (ms)", "X displacement (pixel)", "Y displacement (pixel)", "Radius (pixel)", "Angle"]]
    data_filtered_nopass = data_filtered_nopass[~acceptable_ind]
 
    # Section the data eliminated by the upper bound vs the lower bound
    # lower
    data_filtered_lower_bound_nopass = data_copy.loc[:, ["Time (ms)", "X displacement (pixel)", "Y displacement (pixel)", "Radius (pixel)", "Angle"]]
    data_filtered_lower_bound_nopass = data_filtered_lower_bound_nopass[~down_fil]
   
    # upper
    data_filtered_upper_bound_nopass = data_copy.loc[:, ["Time (ms)", "X displacement (pixel)", "Y displacement (pixel)", "Radius (pixel)", "Angle"]]
    data_filtered_upper_bound_nopass = data_filtered_upper_bound_nopass[~up_fil]

    # _________________________________________[FILTERED DATA]__________________________________________________________
    # Recalculate Delta Angle now with only passing values
    data_filtered_pass["Delta Angle"] = calculate_delta_angle(data_filtered_pass["Angle"])
    # Recalculate Continous now with onyl passing values
    data_filtered_pass["Continuous Angle"] = data_filtered_pass["Delta Angle"].cumsum()

    # For record keeping, set all no passing points to NaN
    # Don't need to calculate the angle for these because they are questionable, so just mark NaN
    data_filtered_nopass["Delta Angle"] = np.nan  
    data_filtered_nopass["Continuous Angle"] = np.nan

    return data, data_filtered_pass, data_filtered_nopass, data_filtered_lower_bound_nopass, data_filtered_upper_bound_nopass



# Give a home to Claire's code: [NOT MOST CURRENT]
# The following function takes a dataframe from the downsampling function and calculates angular data from it via Claire's calculation method, now OUTMODED
# this angle calculation does not have a filter
def avt_no_filter(down_sampled_df, frame_start, frame_end):
    # assign down_sampled_df as df
    df = down_sampled_df

    # DATA PROCESSING FOR COORDINATE CONVERTION ###  CLAIRE CALCULATES ANGLE!
    # theta = (0,360) and thetac =(-infinity degrees, infinity degrees)
    # radian to degree conversion
    r2d = 180/np.pi
    # arctan2 is the full unit circle conversion (-pi,pi) as opposed to (-pi/2,pi/2)

    df_filter = pd.DataFrame(df.iloc[:, 0])
    # find radius
    df_filter['radius'] = np.power(np.power(
        df['Y displacement (pixels)'], 2) + np.power(df['X displacement (pixels)'], 2), 0.5)
    # find theta arctan2 is the full unit circle conversion (-pi,pi) as opposed to (-pi/2,pi/2)
    df_filter['theta'] = - \
        np.arctan2(df['Y displacement (pixels)'],
                   df['X displacement (pixels)'])*r2d
    df_filter['Time (ms)'] = df['Time (ms)']
    # if r is greater than a certain value, the entire row of this dataframe is stored into the next dataframe
    # we conserve the other columns where the row meets the requirement
    df_theta = df_filter.loc[df_filter['radius'] > 0.167].copy()
    # need the .copy() at the end of the line above due to clarity, we want to alter the dataframe to make df_theta, not df_filter
    # arctan2 is the full unit circle conversion (-pi,pi) as opposed to (-pi/2,pi/2)
    # add 360 onto the 3rd and 4th quadrant values to make range from (0,360)
    # df_theta is our (0,360) dataframe
    df_theta.loc[df_theta.theta < 0, ['theta']] += 360

    # make dataframe for angular continuous (base dataframe changes with user preferences)
    # df_theta.iloc[:,2] is the (0,360) theta range
    angularc = pd.DataFrame(df_theta.iloc[:, 2])
    angularc.columns = ['theta']
    print('end')
    # add a row of zeros at the top and reset index
    zero_row = pd.DataFrame({'theta': 0}, index=[0])
    angularc = pd.concat([zero_row, angularc]).reset_index(drop=True)

    # find displacement between rows (row[i+1]-row[i]) 350- 25 == 325 --> -35
    angularc['displacement'] = angularc.diff()  # find displcement between rows
    angularc = angularc.apply(pd.to_numeric, errors='coerce')
    angularc = angularc.dropna()  # drop the NANs if there are any
    angularc = angularc.reset_index(drop=True)  # reset the index
    # store the dataframe into an array
    angular_vector = angularc['displacement'].values
    angular_vectorT = angular_vector.T  # transpose the array into a row vector

    # Now we have displacement between rows
    # if the displacement between two rows is greater than 180, subtract 360 (we assume the rotor went backward)
    # if the displacement between two rows is less than -180, add 360 (we assume the rotor went forward)
    # so we edit the displacement to reflect the rotor movement

    angular_vectorT[angular_vectorT >= (180)] -= 360

    angular_vectorT[angular_vectorT <= (-180)] += 360

    # angular_vectorT[sqrt(x**2+(y)**2) < 0.166] = NaN # get this to work
    # df['Y displacement (pixels)']**2 + df['X displacement (pixels)']**2

    # store it back in a pandas dataframe
    disp = angular_vectorT.T
    cont_rotation = pd.DataFrame(
        disp, columns=['theta displacement correction'])

    # add a row of zeros to the top so we conserve the first row
    zero_row = pd.DataFrame({'theta displacement correction': 0}, index=[0])
    cont_rotation = pd.concat([zero_row, cont_rotation]).reset_index(drop=True)
    # enact a culmulitive sum function that adds together all displacements that came before each row
    cont_rotation['continuous theta'] = cont_rotation.cumsum()
    # drop the NAN and or first row of zeros to start at the actual first data point
    cont_rotation = cont_rotation.apply(pd.to_numeric, errors='coerce')

    cont_rotation = cont_rotation.dropna()
    cont_rotation = cont_rotation.reset_index(drop=True)
    cont_rotation.drop(index=cont_rotation.index[0], axis=0, inplace=True)
    # cont_rotation is our (-infinity,infinity) degree rotation dataframe
    cont_rotation = cont_rotation.reset_index(drop=True)
    # Now we have a dataframe called cont_rotation that has 2 columns
    # first column is displacement with the correction and second column is the culmulitive sum of the first col
    # 'continuous theta' is the cumulitive sum of the displacements

    ## Something to look into ##
    # the assumption there is that even though that jump looks like a backwards jump of ~175 degrees, it’s close enough to 180 degrees that the direction could have been mistaken.
    # and if we are unsure if we are mistaken then let’s look at surrounding frames to get a hint for which direction it is going
    # have to do this after calc theta culmulitive

    # MORE OF CLAIRE CLAIRE'S CALCUATION OF ANGLE AND TIME
    # Assign theta(0,360), time, and theta(-infinity,infinity)-->(continuous degree rotation)
    theta = df_theta.iloc[frame_start:frame_end, 2]
    t = df_theta.iloc[frame_start:frame_end, 3]

    thetac = cont_rotation.iloc[frame_start:frame_end, 1]
    return t, theta, thetac

