import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from pathlib import Path
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import pandas as pd
import DORA2 as DORA
import sma_lib.MAP_Parameters as params

xmlname = "DORA2_settings"

#read in the settings in the .xml file using hazen's Parameter Class
pars = params.Parameters(xmlname+'.xml') #par is an object of type Parameters, defined in sa_library

class MyGUI(QMainWindow):
    
    def __init__(self):
        super(MyGUI,self).__init__()
        uic.loadUi("MAP.1.1.ui",self)
        self.show()

        # Connect the clicked to open folder search
        self.dir_btn.clicked.connect(self.open_dir_dialog) # if button clicked
        
        # set up list box for displaying CSV file names
        self.csv_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.csv_list.setSelectionMode(QListWidget.SingleSelection)

        # Connect the csvclicked in List Box to plotting function
        self.csv_list.itemClicked.connect(self.plot_csv)

        # Conncet the clear selection button with clear csv function
        self.clear_csvs_btn.clicked.connect(self.clear_csv_list)

        # set up figure and canvas
        self.fig_2D = plt.figure()
        self.canvas = FigureCanvas(self.fig_2D)
        self.toolbar = NavigationToolbar(self.canvas,self.centralwidget)
        self.verticalLayout_3.addWidget(self.toolbar)
        self.verticalLayout_3.addWidget(self.canvas)
        

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
            print(type(csv_files))
            for file_name in csv_files:
                self.csv_list.addItem(file_name) 
            print(f'You selected folder {str(path)}')
            
            # print(f'You selected folder {csv_file_paths}')
            

    def plot_csv(self):
        # clear previous plot
        self.fig_2D.clf()

        # get selected CSV file name and path
        selected_csv = self.csv_list.currentItem().text()
        dir_path = self.dir_path

        
        # Load the csv into a data frame and remove NaN's
        data, __ , __ = DORA.load_csv(selected_csv,dir_path)
        
        # Center the data via center finding algorithm
        center, radius_estimate = DORA.find_center(data)

        data = DORA.calculate_time_angle(data,center)
        if pars.processing == "downsample":
            down_sampled_df,frame_start,frame_end = DORA.downsample(data)
        else:
            down_sampled_df = DORA.downsample(data)

        # DORA.plot_2D_graph(down_sampled_df, fig = self.fig_2D, ax = self.ax_2D)
        DORA.plot_2D_graph(down_sampled_df, title = selected_csv, fig = self.fig_2D)

        self.fig_2D.tight_layout()
        # redraw canvas
        self.canvas.draw()

    def clear_csv_list(self):
        self.csv_list.clear()
        self.dir_name_edit.clear()
        print("You have cleared the CSV's")

        
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