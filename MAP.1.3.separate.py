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

xmlname = "DORA2_settings"

#read in the settings in the .xml file using hazen's Parameter Class
pars = params.Parameters(xmlname+'.xml') #par is an object of type Parameters, defined in sa_library

class MyGUI(QMainWindow):
    
    def __init__(self):
        super(MyGUI,self).__init__()
        uic.loadUi("MAP.1.3.ui",self)
        self.show()

        # Connect the clicked to open folder search
        self.dir_btn.clicked.connect(self.open_dir_dialog) # if button clicked
        
        # set up list box for displaying CSV file names
        self.csv_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.csv_list.setSelectionMode(QListWidget.SingleSelection)

        # Connect the csvclicked in List Box to plotting function
        self.csv_list.itemClicked.connect(self.load_choosen_csv)
        self.csv_list.itemSelectionChanged.connect(self.load_choosen_csv)

        # Connect the clear selection button with clear csv function
        self.clear_csvs_btn.clicked.connect(self.clear_csv_list)

        # set up figure and canvas for Intensity graph
        self.fig_2 = plt.figure()
        self.canvas_2 = FigureCanvas(self.fig_2)
        self.toolbar_2 = NavigationToolbar(self.canvas_2,self.centralwidget)
        self.graphLayout_2.addWidget(self.toolbar_2)
        self.graphLayout_2.addWidget(self.canvas_2)

        # set up figure and canvas for 2D graph
        self.fig_2D = plt.figure()
        self.canvas_2D = FigureCanvas(self.fig_2D)
        self.toolbar_2D = NavigationToolbar(self.canvas_2D,self.centralwidget)
        self.graphLayout_2D.addWidget(self.toolbar_2D)
        self.graphLayout_2D.addWidget(self.canvas_2D)
        
        # Intialize value of Line Edit:
        self.frame_increment_LineEdit.setText(str(pars.frame_increment))

        # Connect the frame_slider_increment_LineEditor to the LineEdit
        self.update_frames_intensity_PushButton.clicked.connect(self.update_input_values)
        # Connect the frame_slider to update_2D_plot
        self.frame_slider.valueChanged.connect(self.update_2D_plot)

        #set default function values
        self.frame_increment = pars.frame_increment
        


    def open_dir_dialog(self):
        # open file dialog and get directory
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.dir_path = dir_path         
        if dir_path:
            path = Path(dir_path)
            
            #update text
            self.dir_name_edit.setText(str(path))

            #populate CSV list
            csv_files, _ = get_csv_files(str(path))
            for file_name in csv_files:
                self.csv_list.addItem(file_name) 
            print(f'[INFO] You selected folder {str(path)}')
            
            # print(f'You selected folder {csv_file_paths}')
            

    def load_choosen_csv(self):

        # get selected CSV file name and path
        selected_csv = self.csv_list.currentItem().text()
        self.selected_csv = selected_csv
        dir_path = self.dir_path

        # Load the csv into a data frame
        self.raw_data = DORA.load_csv(selected_csv,dir_path)

        # Intialize bounds 
        self.start_frame = self.raw_data["index"].min()
        self.end_frame = self.raw_data["index"].max()
        self.min_intensity = self.raw_data["Intensity"].min()
        self.max_intensity = self.raw_data["Intensity"].max()

        # Update the GUI to reflect it 
        self.frame_start_LineEdit.setText(str(self.start_frame))
        self.frame_end_LineEdit.setText(str(self.end_frame))
        self.intensity_min_LineEdit.setText(str(self.min_intensity))
        self.intensity_max_LineEdit.setText(str(self.max_intensity))

        # Continue DORA processing
        self.raw_data, __, __ = DORA.remove_invalid_readings(self.raw_data)
        self.center, __ = DORA.find_center(self.raw_data)
        
        #Add first found center into center_list 
        self.center_list = np.array(self.center)
        print(f"my starting center is {self.center_list}")
        
        # Update Center Label
        self.center_x_LineEdit.setText(str(0)) 
        self.center_y_LineEdit.setText(str(0)) 
        
        # Continue DORA
        self.centered_data = DORA.generate_centered_data(self.raw_data, self.center)
        self.centered_data = DORA.calculate_angle(self.centered_data)
        if pars.processing == "downsample":
            down_sampled_df,frame_start,frame_end = DORA.downsample(self.centered_data)
            # self.frame_start = frame_start
            # self.frame_end = frame_end
        else:
            down_sampled_df = DORA.downsample(self.centered_data)

        # Store the data
        self.data = down_sampled_df

        # change to np format
        self.data_array = self.data.to_numpy()

        # set minimums and maximums for frame slider values 
        data_max = self.data_array.shape[0]
        self.frame_slider.setMaximum(data_max - self.frame_increment)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(self.frame_slider.minimum())
        
        # Update the graphs
        self.update_graph_panel()

    def update_graph_panel(self):
        # clear previous plot
        self.fig_2D.clf()
        
        # 2D graph
        DORA.plot_2D_graph(self.data, title = self.selected_csv, fig = self.fig_2D)
        self.fig_2D.tight_layout()

        # Angle vs Time Graph
        DORA.plot_angular_continuous(self.data, title = self.selected_csv, fig = self.fig_2)
        self.fig_2.tight_layout()

        # redraw canvas
        self.canvas_2.draw()
        self.canvas_2D.draw()

    def update_input_values(self):
        
        # Check if inputs are numeric
        lineEdits = [ 
            self.frame_increment_LineEdit, 
            self.center_x_LineEdit, 
            self.center_y_LineEdit, 
            self.frame_start_LineEdit, 
            self.frame_end_LineEdit, 
            self.intensity_min_LineEdit, 
            self.intensity_max_LineEdit
        ]

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
            self.raw_data, __, __ = DORA.remove_invalid_readings(self.raw_data)
            # self.center, __ = DORA.find_center(self.raw_data)
            self.centered_data = DORA.generate_centered_data(self.raw_data, self.center)
            self.centered_data = DORA.calculate_angle(self.centered_data)
            if pars.processing == "downsample":
                self.data ,frame_start,frame_end = DORA.downsample(self.centered_data)
            else:
                self.data = DORA.downsample(self.centered_data)

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

        self.fig_2D.tight_layout()

        # redraw canvas
        self.canvas_2D.draw()

    def clear_csv_list(self):
        self.csv_list.clear()
        self.dir_name_edit.clear()
        print("You have cleared the CSV's")

def update_value(self, line_edit, attribute_name):
    value = line_edit.text()
    try:
        if attribute_name == "start_frame" or attribute_name == "end_frame" or attribute_name == "frame_increment":
            setattr(self, attribute_name, int(float(value)))
        else:
            setattr(self, attribute_name, float(value))
        print(f"[INFO] New {attribute_name} is {getattr(self, attribute_name)}")
    except ValueError:
        print(f"[ERROR] Invalid input: {value} for {attribute_name}")

def check_bounds(lower_bound, upper_bound, label):

    if upper_bound == -1:
        return True
    
    if lower_bound < -1 or upper_bound < -1:
        print(f"[INFO] Invalid {label} bounds! Bounds should not negative, unless negative -1 on the upperbound.")
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
    '''Given folder path, output lists of csv files and csv file paths'''
    files = os.listdir(folder_path)
    # Filter the list to only include CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    csv_file_paths = [os.path.join(folder_path, file) for file in files if file.lower().endswith('.csv')]
    return csv_files, csv_file_paths   

def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()
    pass

if __name__ == '__main__':
    main()