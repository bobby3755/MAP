'''
This module contains the PyQT5 circutry that runs the GUI. GUI structure is read in from a UI file. DORA parameters are read in a DORA2_settings XML.
'''
import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from pathlib import Path
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import sma_lib.DORA2 as DORA
import sma_lib.MAP_Parameters as params

# xmlname = "DORA2_settings"

# #read in the settings in the .xml file using hazen's Parameter Class
# pars = params.Parameters(xmlname+'.xml') #par is an object of type Parameters, defined in sa_library

xmlname = "DORA2_settings.xml"  # Update the XML filename

# Construct the absolute path to the XML file
directory_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(directory_path, xmlname)

# Read in the settings from the XML file using the Parameters class
pars = params.Parameters(xml_path)  # Pass the updated XML file path

class MyGUI(QMainWindow):
    """
    A custom GUI class for handling data visualization and analysis.

    Attributes:
        dir_path (str): The selected directory path.
        selected_csv (str): The currently selected CSV file name.
        raw_data (DataFrame): The raw data loaded from the CSV file.
        data (DataFrame): The processed data after removing invalid readings.
        center (ndarray): The center coordinates of the data.
        center_list (ndarray): A running list of center manipulations.
        frame_increment (int): The number of frames to display in each update.
        start_frame (int): The starting frame index.
        end_frame (int): The ending frame index.
        min_intensity (float): The minimum intensity threshold.
        max_intensity (float): The maximum intensity threshold.
        fig_2 (Figure): The figure object for the Intensity graph.
        canvas_2 (FigureCanvas): The canvas to display the Intensity graph.
        fig_2D (Figure): The figure object for the 2D graph.
        canvas_2D (FigureCanvas): The canvas to display the 2D graph.
        choosen_secondary_graph (str): The currently selected secondary graph option.

    Methods:
        open_dir_dialog(): Opens a file dialog to select a directory.
        load_choosen_csv(): Loads the selected CSV file and performs data processing.
        update_graph_panel(): Updates the graph panel with the latest data.
        update_input_values(): Updates the input values based on the user's inputs.
        update_2D_plot(): Updates the 2D graph based on the frame slider value.
        update_secondary_graph(): Updates the secondary graph based on the selected option.
        clear_csv_list(): Clears the CSV file list.

    """


    def __init__(self):
        """
        Initializes an instance of the MyGUI class.

        This method sets up the GUI elements, connects them to the corresponding functions,
        and initializes default values for certain attributes.

        Note:
            The structure of the GUI is read in from a UI file ("MAP.1.3.ui") in this method.

        Parameters:
            None

        Returns:
            None
        """
        super(MyGUI,self).__init__()
        uic.loadUi("MAP.1.3.color.ui",self)
        self.show()

        # Connect the clicked to open folder search
        self.dir_btn.clicked.connect(self.open_dir_dialog) # if button clicked
        
        # set up list box for displaying CSV file names
        self.csv_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.csv_list.setSelectionMode(QListWidget.SingleSelection)

        # Connect the csvclicked in List Box to plotting function
        self.csv_list.itemClicked.connect(self.load_choosen_csv)
        self.csv_list.currentItemChanged.connect(self.load_choosen_csv)

        # Connect the clear selection button with clear csv function
        self.clear_csvs_btn.clicked.connect(self.clear_csv_list)

        # set up figure and canvas for Intensity graph
        self.fig_2 = plt.figure()
        self.canvas_2 = FigureCanvas(self.fig_2)
        self.toolbar_2 = NavigationToolbar(self.canvas_2,self.centralwidget)
        self.graphLayout_2.addWidget(self.toolbar_2)
        self.graphLayout_2.addWidget(self.canvas_2)
        

        # set up figure and canvas for 2D graph
        # self.graphLayout_2D.setMinimumSize(480, 528)
        self.fig_2D = plt.figure()
        self.canvas_2D = FigureCanvas(self.fig_2D)
        self.toolbar_2D = NavigationToolbar(self.canvas_2D,self.centralwidget)
        self.graphLayout_2D.addWidget(self.toolbar_2D)
        self.graphLayout_2D.addWidget(self.canvas_2D)

        # Connect secondary_graph_comboBox to change secondary graph
        self.secondary_graph_comboBox.activated.connect(self.update_secondary_graph)
        
        # Intialize value of Line Edit:
        self.frame_increment_LineEdit.setText(str(pars.frame_increment))

        # Connect the frame_slider_increment_LineEditor to the LineEdit
        self.update_frames_intensity_PushButton.clicked.connect(self.update_input_values)

        # Connect the frame_slider to update_2D_plot
        self.frame_slider.valueChanged.connect(self.update_2D_plot)

        #set default function values
        self.frame_increment = pars.frame_increment
        secondary_graph_options = ["Continuous Angle vs Time", "Intensity vs Time"]
        for option in secondary_graph_options:
            self.secondary_graph_comboBox.addItem(option)
        self.choosen_secondary_graph = self.secondary_graph_comboBox.currentText()

    def open_dir_dialog(self):
        """
        Opens a file dialog to select a directory and performs necessary operations based on the selected directory.

        This method prompts the user to select a directory using a file dialog and performs the following tasks:
        - Updates the directory path attribute of the GUI.
        - Updates the directory name in the corresponding text field of the GUI.
        - Populates the CSV list with file names from the selected directory.
        - Prints an information message indicating the selected folder.

        Parameters:
            None

        Returns:
            None
        """
        # Open file dialog and get directory
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.dir_path = dir_path

        if dir_path:
            path = Path(dir_path)

            # Update text field with directory name
            self.dir_name_edit.setText(str(path))

            # Populate CSV list
            csv_files, _ = get_csv_files(str(path))
            for file_name in csv_files:
                self.csv_list.addItem(file_name)

            print(f'[INFO] You selected folder {str(path)}')

    def load_choosen_csv(self):
        """
        Loads the chosen CSV file, performs data processing, and updates the GUI.

        This method retrieves the selected CSV file from the CSV list, runs the necessary data processing steps using the DORA pipeline,
        and updates the GUI with the processed data. It sets up the frame slider and updates the graphs to reflect the loaded data.

        Parameters:
            None

        Returns:
            None
        """
        # Get selected CSV file name and path
        selected_csv = self.csv_list.currentItem().text()
        self.selected_csv = selected_csv
        dir_path = self.dir_path

        # Run DORA pipeline
        self.raw_data = DORA.load_csv(selected_csv, dir_path)
        self.data = self.raw_data.copy()  # Copy data so raw data remains intact
        DORA.remove_invalid_readings(self.data)
        DORA.remove_nopass_intensity_filter(self.data)
        self.center, __ = DORA.find_center(self.data)
        self.data["X displacement (pixel)"], self.data["Y displacement (pixel)"] = DORA.generate_centered_data(self.data, self.center)
        self.data['Radius (pixel)'] = DORA.calculate_radius(self.data)
        self.data['z-score Rad'] = DORA.calculate_rad_zscore(self.data)
        self.data["Angle"], self.data["Delta Angle"], self.data["Continuous Angle"] = DORA.calculate_angle(self.data)
        down_sampled_df = DORA.downsample(self.data.dropna())

        # Initialize frame and intensity bounds
        self.start_frame = self.raw_data["index"].min()
        self.end_frame = self.raw_data["index"].max()
        self.min_intensity = self.raw_data["Intensity"].min()
        self.max_intensity = self.raw_data["Intensity"].max()

        # Update the GUI to reflect new frame and intensity bounds
        self.frame_start_LineEdit.setText(str(self.start_frame))
        self.frame_end_LineEdit.setText(str(self.end_frame))
        self.intensity_min_LineEdit.setText(str(self.min_intensity))
        self.intensity_max_LineEdit.setText(str(self.max_intensity))

        # Add first found center into center_list
        self.center_list = np.array(self.center)
        print(f"my starting center is {self.center_list}")

        # Update Center Label
        self.center_x_LineEdit.setText(str(0))
        self.center_y_LineEdit.setText(str(0))

        # Store the data
        self.data = down_sampled_df

        # Change to np format
        self.data_array = self.data.to_numpy()

        # Set minimums and maximums for frame slider values
        data_max = self.data_array.shape[0]
        self.frame_slider.setMaximum(data_max - self.frame_increment)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(self.frame_slider.minimum())

        # Update the graphs
        self.update_graph_panel()

    def update_graph_panel(self):
        """
        Updates the graph panel by clearing the previous plot and generating a new 2D graph based on the loaded data.

        This method clears the previous plot in the graph panel, generates a new 2D graph using the DORA plot_2D_graph function,
        and updates the canvas to display the updated graph. It also calls the update_secondary_graph method to update the secondary graph.

        Parameters:
            None

        Returns:
            None
        """
        # Clear the previous plot
        self.fig_2D.clf()

        # Generate the 2D graph
        DORA.plot_2D_graph(self.data, title=self.selected_csv, fig=self.fig_2D)
        self.fig_2D.tight_layout()

        # Update the canvas to display the updated graph
        self.canvas_2D.draw()

        # Update the secondary graph
        self.update_secondary_graph()      

    def update_input_values(self):
        """
        Validates and updates the input values from the line edit fields.

        This method retrieves the values from the line edit fields and performs several checks and actions:

        Checks:
            - Checks if all input values are numeric. If any value is not numeric, an error message is printed, and the
            method returns without updating the values.
            - Checks if the minimum and maximum bounds for frames and intensity values are valid. Uses the `check_bounds`
            function to perform the check.

        Actions:
            - Updates the attribute values in the class corresponding to the line edit fields.
            - Converts the chosen x and y center coordinates from nm to pixels if the unit is "nm".
            - Saves the center coordinates in pixels and checks if the center is redundant. If it's new, adds it to the
            running list of centers.
            - Updates the start frame, end frame, minimum intensity, and maximum intensity attributes.
            - Retrieves the selected CSV file name and path.
            - Loads the CSV file into a data frame, removes invalid readings, and applies intensity filtering.
            - Generates centered data based on the updated center coordinates.
            - Calculates radius, z-score rad, angle, delta angle, and continuous angle for the data.
            - Downsamples the data by removing NaN values.
            - Converts the data frame to a NumPy array.
            - Updates the frame slider's minimum and maximum values based on the data.
            - Updates the graphs by clearing the previous plot and plotting the updated 2D graph.
            
        Parameters:
            None

        Returns:
            None
        """
        # Create a list of all line edits
        lineEdits = [ 
            self.frame_increment_LineEdit, 
            self.center_x_LineEdit, 
            self.center_y_LineEdit, 
            self.frame_start_LineEdit, 
            self.frame_end_LineEdit, 
            self.intensity_min_LineEdit, 
            self.intensity_max_LineEdit
        ]
        
        # Check if inputs are numeric
        for lineEdit in lineEdits:
            value = lineEdit.text()
            try:
                float(value)  # Try converting the value to float
            except ValueError:
                print(f"This value input is not numeric: {value}")
                return

        # Check if minimum and maximum bounds make sense. check_bounds outputs True if it passes
        check_1 = check_bounds(float(self.frame_start_LineEdit.text()), float(self.frame_end_LineEdit.text()), label = "Frames")
        check_2 = check_bounds(float((self.intensity_min_LineEdit.text())), float(self.intensity_max_LineEdit.text()), label = "Intensity")

        if all(value == True for value in [check_1,check_2]):

            #update the attributes values in the class corresponding to the line edits 
            update_value(self, self.frame_increment_LineEdit, "frame_increment")
            update_value(self, self.center_x_LineEdit, "center_x")
            update_value(self, self.center_y_LineEdit, "center_y")
            if pars.unit == "nm":
                #convert choosen x (nm) and y (nm) into pixels
                self.center_x = self.center_x / pars.pixel_size
                self.center_y = self.center_y / pars.pixel_size
            
            # save center in pixels
            self.added_center = [self.center_x, self.center_y]

            # check if the center is redundant
            if self.added_center not in self.center_list:
                # add center if it is new
                self.center_list = np.vstack((self.center_list, self.added_center)) # adds a running list of center manipulations so that all centering opperations are relative to the original data
            self.center = np.sum(self.center_list, axis=0)
            
            print(f"[INFO] Adjusted Center in pixel space is {self.center}")
            update_value(self, self.frame_start_LineEdit, "start_frame")
            update_value(self, self.frame_end_LineEdit, "end_frame")
            update_value(self, self.intensity_min_LineEdit, "min_intensity")
            update_value(self, self.intensity_max_LineEdit, "max_intensity")

            # get selected CSV file name and path
            selected_csv = self.csv_list.currentItem().text()
            self.selected_csv = selected_csv
            dir_path = self.dir_path

            # Load the csv into a data frame and remove NaN's
            self.raw_data = DORA.load_csv(selected_csv, dir_path, start_frame = self.start_frame, end_frame = self.end_frame)
            self.data = self.raw_data.copy() # Copy data so raw data remains intact
            DORA.remove_invalid_readings(self.data)
            DORA.remove_nopass_intensity_filter(self.data, min_intensity= self.min_intensity, max_intensity=self.max_intensity) #!! Change
            self.data["X displacement (pixel)"], self.data["Y displacement (pixel)"] = DORA.generate_centered_data(self.data, self.center)
            self.data['Radius (pixel)'] = DORA.calculate_radius(self.data)
            self.data['z-score Rad'] = DORA.calculate_rad_zscore(self.data)
            self.data["Angle"], self.data["Delta Angle"], self.data["Continuous Angle"] = DORA.calculate_angle(self.data)
            down_sampled_df = DORA.downsample(self.data.dropna())
        
            # change to np format
            self.data_array = self.data.to_numpy()

            # update the frame_slider to have new bounds
            frame_slider_index_min = np.min(self.data_array[:, 0])
            frame_slider_index_max = np.max(self.data_array[:,0])
            self.frame_slider.setMinimum(int(frame_slider_index_min))
            self.frame_slider.setMaximum(int(frame_slider_index_max)-self.frame_increment)
            self.frame_slider.setValue(self.frame_slider.minimum())

            # Update the graphs
            self.update_graph_panel()

    def update_2D_plot(self):
        # load slider value INDEXING VALUE
        # NOTE: you need to subtract the minimum value otherwise you will index more than you need to when min ~= 0 
        slider_index = self.frame_slider.value() - self.frame_slider.minimum()

        # Index a subset to graph
        subset_array = self.data_array[slider_index : slider_index + self.frame_increment,:]

        # Get the minimum index value from the first column
        min_index = np.min(subset_array[:, 0])
        max_index = np.max(subset_array[:,0])

        # Graph
        # ... perform your 2D graph plotting using subset_array ...
        # ... replace the following code with your actual graph plotting code ...
        DORA.plot_2D_graph(subset_array, column_headers= self.data.columns, start_frame= min_index, endframe = max_index, title= self.selected_csv, fig = self.fig_2D)
        # self.fig_2D.tight_layout()

        # redraw canvas
        self.canvas_2D.draw()

    def update_secondary_graph(self):
        """
        Updates the secondary graph based on the selected option from the combo box.

        This method performs the following actions:

        - Clears the previous plot on the secondary graph canvas.
        - Retrieves the selected option from the secondary graph combo box and updates the label text accordingly.
        - If the selected option is "Continuous Angle vs Time", plots the angular continuous graph using the DORA
        `plot_angular_continuous` function.
        - If the selected option is "Intensity vs Time", plots the intensity vs. time graph using the DORA
        `plot_intensity_time` function.
        - If the selected option is not part of the approved list, raises a ValueError.

        Parameters:
            None

        Returns:
            None
        """

        # Reset Graph Canvas
        self.fig_2.clf()

        # Redefine choosen secondary graph
        self.choosen_secondary_graph = self.secondary_graph_comboBox.currentText()
        self.label_2.setText(self.choosen_secondary_graph)

        # If specific graph is choose, plot that graph
        if self.choosen_secondary_graph == "Continuous Angle vs Time":
            DORA.plot_angular_continuous(self.data, title = self.selected_csv, fig = self.fig_2)

        elif self.choosen_secondary_graph == "Intensity vs Time":
            DORA.plot_intensity_time(self.data, title = self.selected_csv, fig = self.fig_2) 
    
        else:
            ValueError("[ERROR] The choosen secondary graph combo box selection is NOT part of the approved list")

        # Apply tight layout
        self.fig_2.tight_layout()

        # redraw canvas
        self.canvas_2.draw()

    def clear_csv_list(self):
        """
        Clears the CSV list and directory name edit.

        This method performs the following actions:

        - Clears the items in the CSV list widget.
        - Clears the text in the directory name edit widget.
        - Prints a message indicating that the CSVs have been cleared.

        Parameters:
            None

        Returns:
            None
        """
        self.csv_list.clear()
        self.dir_name_edit.clear()
        print("You have cleared the CSV's")

def update_value(self, line_edit, attribute_name):
    """
    Updates the value of the specified attribute based on the input from a line edit.

    Args:
        line_edit (QLineEdit): The line edit containing the new value.
        attribute_name (str): The name of the attribute to be updated.

    Returns:
        value: The updated attribute value.

    Notes:
        - This method takes the input from a line edit and assigns it to the corresponding attribute.
        - The naming convention for the associated line edits is: parameter_LineEdit.
        - The method converts the input value to the appropriate data type (integer or float) based on the attribute.
        - It assumes that these attributes are only numeric.
    """
    value = line_edit.text()
    try:
        if attribute_name == "start_frame" or attribute_name == "end_frame" or attribute_name == "frame_increment":
            setattr(self, attribute_name, int(float(value))) # Convert to integer if necessary
        else:
            setattr(self, attribute_name, float(value))  # Convert to float
        print(f"[INFO] New {attribute_name} is {getattr(self, attribute_name)}")
    except ValueError:
        print(f"[ERROR] Invalid input: {value} for {attribute_name}")

def check_bounds(lower_bound, upper_bound, label):
    """
    Checks the validity of lower and upper bounds for a given label.

    Args:
        lower_bound (float): The lower bound value.
        upper_bound (float): The upper bound value.
        label (str): The label or parameter associated with the bounds.

    Returns:
        bool: True if the bounds are valid, False otherwise.

    Notes:
        - This function is used to check the validity of bounds for a specific parameter.
        - The lower_bound and upper_bound are checked against certain conditions.
        - If the bounds are valid, the function returns True; otherwise, it returns False.
        - It also prints informative messages about the validity of the bounds.
    """
    if upper_bound == -1:
        return True

    if lower_bound < -1 or upper_bound < -1:
        print(f"[INFO] Invalid {label} bounds! Bounds should not be negative, unless -1 is used for the upper bound.")
        return False

    if upper_bound < lower_bound:
        print(f"[INFO] Invalid {label} bounds! The upper bound should be greater than or equal to the lower bound.")
        return False
    elif lower_bound == upper_bound:
        print(f"[INFO] Invalid {label} bounds! The lower bound and upper bound should be different.")
        return False
    else:
        print(f"[INFO] Valid {label} bounds entered.")
        return True

        
def get_csv_files(folder_path):
    """
    Retrieves CSV files and their file paths from a given folder path.

    Args:
        folder_path (str): The path to the folder containing CSV files.

    Returns:
        tuple: A tuple containing two lists:
            - csv_files: A list of CSV file names in the folder.
            - csv_file_paths: A list of full paths to the CSV files.

    Note:
        - This function uses the `os` module to retrieve the list of files in the folder.
        - It filters the list to include only CSV files based on their file extensions.
        - The function returns two lists, one with the file names and the other with the full paths.
    """
    files = os.listdir(folder_path)
    # Filter the list to only include CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    csv_file_paths = [os.path.join(folder_path, file) for file in files if file.lower().endswith('.csv')]
    return csv_files, csv_file_paths  

def main():
    '''
    The main function represents the entry point of your application. It initializes the application, creates the main GUI window, and starts the event loop.
    '''
    app = QApplication([])
    window = MyGUI()
    app.exec_()
    pass

if __name__ == '__main__':
    main()