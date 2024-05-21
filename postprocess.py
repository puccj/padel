import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# TO DO: make this a class

def create_heatmaps(input_csv_path, output_path = "output_heatmaps/"):

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
            # Each row is a list where each element represents a cell in that row
            coordinates_1 = tuple(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', row[2])))
            coordinates_2 = tuple(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', row[6])))
            coordinates_3 = tuple(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', row[10])))
            coordinates_4 = tuple(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', row[14])))
            player1_positions.append(coordinates_1)
            player2_positions.append(coordinates_2)
            player3_positions.append(coordinates_3)
            player4_positions.append(coordinates_4)
        
    pos1 = np.array(player1_positions)
    pos2 = np.array(player2_positions)
    pos3 = np.array(player3_positions)
    pos4 = np.array(player4_positions)


    # Background
    court_img = mpimg.imread('Field.png')  # Replace 'tennis_court_image.jpg' with the path to your image
    plt.figure(facecolor=(0.54,0.73,0.82))
    plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image

    # Plot player positions with transparency (alpha=0.5)
    plt.scatter(pos1[:, 0], pos1[:, 1], color='red', alpha=0.5, label='Player 1')

    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig(output_path + 'heatmap_1.png',bbox_inches='tight')


    plt.figure(facecolor=(0.54,0.73,0.82))
    plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image

    # Plot player positions with transparency (alpha=0.5)
    plt.scatter(pos2[:, 0], pos2[:, 1], color='green', alpha=0.5, label='Player 2')

    plt.xticks([])
    plt.yticks([])

    plt.legend()

    plt.savefig(output_path + 'heatmap_2.png',bbox_inches='tight')

    
    plt.figure(facecolor=(0.54,0.73,0.82))
    plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image

    # Plot player positions with transparency (alpha=0.5)
    plt.scatter(pos3[:, 0], pos3[:, 1], color='blue', alpha=0.5, label='Player 3')

    plt.xticks([])
    plt.yticks([])

    plt.legend()

    plt.savefig(output_path + 'heatmap_3.png',bbox_inches='tight')


    plt.figure(facecolor=(0.54,0.73,0.82))
    plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image

    # Plot player positions with transparency (alpha=0.5)
    plt.scatter(pos4[:, 0], pos4[:, 1], color='yellow', alpha=0.5, label='Player 4')

    plt.xticks([])
    plt.yticks([])

    plt.legend()

    plt.savefig(output_path + 'heatmap_4.png',bbox_inches='tight')