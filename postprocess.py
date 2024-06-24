import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from padel_utils import ensure_directory_exists

# TODO: make this a class

def create_heatmaps(input_csv_path, output_path = None, alpha=0.05):

    # check if input csv file contains data (more than 1 row). If non-existent, an error is raised
    with open(input_csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        if len(list(csvreader)) < 2:
            raise ValueError(f"POSTPROCESS ERROR: Input CSV file '{input_csv_path}' does not contain any data.")
    
    if output_path is None:
        output_path = input_csv_path.split('.')[0] + '-HM'

    ensure_directory_exists(output_path)

    player1_positions = []
    player2_positions = []
    player3_positions = []
    player4_positions = []

    with open(input_csv_path, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        
        # Skip the header row
        next(csvreader)
        
        # Iterate over each row in the CSV file
        for row in csvreader:
            # Add positions only if ID is present, otherwise position (0,0) is fictitious
            if row[1]:
                player1_positions.append([float(x) for x in row[2].strip('[]').split()])
            if row[5]:
                player2_positions.append([float(x) for x in row[6].strip('[]').split()])
            if row[9]:
                player3_positions.append([float(x) for x in row[10].strip('[]').split()])
            if row[13]:
                player4_positions.append([float(x) for x in row[14].strip('[]').split()])
        
    pos1 = np.array(player1_positions)
    pos2 = np.array(player2_positions)
    pos3 = np.array(player3_positions)
    pos4 = np.array(player4_positions)


    # Background
    court_img = mpimg.imread('Field.png')
    bg_color = (0.54,0.73,0.82)

    if len(pos1) == 0:
        print("No position data for player 1")
    else:
        plt.figure(facecolor=bg_color)
        plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image
        plt.scatter(pos1[:, 0], pos1[:, 1], color='red', alpha=alpha, label='Player 1')
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.savefig(output_path + 'player1.png',bbox_inches='tight')

    if len(pos2) == 0:
        print("No position data for player 2")
    else:
        plt.figure(facecolor=bg_color)
        plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image
        plt.scatter(pos2[:, 0], pos2[:, 1], color='green', alpha=alpha, label='Player 2')
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.savefig(output_path + 'player2.png',bbox_inches='tight')

    if len(pos3) == 0:
        print("No position data for player 3")
    else:
        plt.figure(facecolor=bg_color)
        plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image
        plt.scatter(pos3[:, 0], pos3[:, 1], color='blue', alpha=alpha, label='Player 3')
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.savefig(output_path + 'player3.png',bbox_inches='tight')

    if len(pos4) == 0:
        print("No position data for player 4")
    else:
        plt.figure(facecolor=bg_color)
        plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image
        plt.scatter(pos4[:, 0], pos4[:, 1], color='yellow', alpha=alpha, label='Player 4')
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.savefig(output_path + 'player4.png',bbox_inches='tight')