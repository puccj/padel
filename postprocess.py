import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter

from padel_utils import ensure_directory_exists

# TODO: make this a class

def read_csv(input_csv_path):
    """ 
    Read a csv file and return a list of dictionaries (one per frame), each containing {'frame_num', 'players'}.
    'players' is a list of dictionaries (one per player), each is {'id', 'position'}
    Input csv file is expected to have the following format:
    frame_num, player0_id, player0_position, player1_id, player1_position, ... , playerX_id, playerX_position
    """
    
    organized_data = []

    with open(input_csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        if len(list(csvreader)) < 1:
            raise ValueError(f"POSTPROCESS ERROR: Input CSV file '{input_csv_path}' does not contain any data.")
        
        for row in csvreader:
            frame_num = int(row[0])
            player_data = row[1:]
            
            players = []
            for i in range(0, len(player_data), 2):
                player_id = int(player_data[i])
                player_position = player_data[i + 1]
                players.append({'id': player_id, 'position': player_position})
            
            frame_data = {'frame_num': frame_num, 'players': players}
            organized_data.append(frame_data)
    
    return organized_data

def analyze_data(all_data):
    """
    Analyzes the data and returns the position of 4 players in each frame 
    The input data is a list of dictionaries (one per frame), each is {'frame_num', 'players'}. 
    'players' is a list of dictionaries (one per player), each is {'id', 'position'}
    """

    # list of 4 sets of IDs, one set for each player
    # selected_ids_list = [set() for _ in range(4)]

    # list of 4 sets of available IDs, one set for each player
    # available_ids_list = [set() for _ in range(4)]

    id_counter = Counter()

    for frame_data in all_data:
        for player in frame_data['players']:
            id_counter[player['id']] += 1
    available_ids_list = [set(id_counter.keys()) for _ in range(4)] # 4 copies of all IDs

    # Get the 4 most common IDs and use to initializa the selected_ids sets
    # TODO: Warning: it could happen that two of the most common IDs are the same player.
    #       For now I'll just check if all 4 most common IDs are simultaneously present at least in one frame 
    #       (but do nothing if that is the case)
    #       Maybe I just need to check the 5th most common ID, excluding one of the previous 4 (trying with all combination) 
    #       and check again if these new 4 IDs are simultaneous in one frame.  And so on cheching the 6th, 7th, etc. most common IDs
    most_common_ids = [id for id, _ in id_counter.most_common(4)]

    # Check if all 4 most common IDs are present in the same frame
    all_together = False
    for frame_data in all_data:
        frame_ids = [player['id'] for player in frame_data['players']]
        if all(id in frame_ids for id in most_common_ids):
            all_together = True
            break
    if not all_together:
        raise NotImplementedError("POSTPROCESS NOT_IMPLEMENTED: The 4 most common IDs are not present in a same frame. Unsure if they belongs to different players. Aborting...")
    
    selected_ids_list = [set([most_common_ids[i]]) for i in range(4)]
    for available_ids in available_ids_list:
        available_ids.difference_update([most_common_ids])  # Assigned IDs are not available for anyone

    #TODO: Make this a function
    # If an ID is present in the same frame as the ID representing a player, it is descarded for that player, since it can't be the same player
        # for frame_data in all_data:
        #     frame_players_ids = [player['id'] for player in frame_data['players']]
        #     for i, selected_ids in enumerate(selected_ids_list):
        #         for selected_id in selected_ids:
        #             if selected_id in frame_players_ids:
        #                 available_ids_list[i].difference_update([id for id in frame_players_ids if id != selected_id])
    for frame_data in all_data:
        frame_players_ids = [player['id'] for player in frame_data['players']]
        for i, selected_id in enumerate(most_common_ids):
            if selected_id in frame_players_ids:
                # available_ids_list[i].difference_update([id for id in frame_players_ids if id != selected_id])
                available_ids_list[i].difference_update([id for id in frame_players_ids])

    # Check if there are still available IDs for each player. If so, add them to the selected_ids (and remove from available)
    
    #while there are element in at least one of the available_ids_list
    while any(available_ids_list):
        multiple_ids = set()    # will contain the IDs present in more than one available_ids list
        added_ids = [None]*4    # will contain the ID added for each Player
        player_to_take = -1     # index of the player that will take the ID

        while not any(added_ids):
            one_is_long_enough = False
            player_to_take += 1

            for i, available_ids in enumerate(available_ids_list):    #for each Player
                if len(available_ids) <= player_to_take:
                    print(f"Player {i} has not enough available IDs to check the {player_to_take}th most common ID")
                    continue
                one_is_long_enough = True
                sorted_id = sorted(available_ids, key = lambda id: id_counter[id], reverse=True)[player_to_take]
                # sorted_ids[0]   #take the most common id available

                in_other_list = False
                for j in range(len(available_ids_list)):   #see if it is contained in other lists
                    if i != j and sorted_id in available_ids_list[j]:
                        multiple_ids.add(sorted_id)
                        in_other_list = True
                        break
            
                if not in_other_list:
                    selected_ids_list[i].add(sorted_id)
                    available_ids_list[i].remove(sorted_id)     # (for other available_ids, the ID is not available as already checked)
                    added_ids[i] = sorted_id
            
            if not one_is_long_enough:
                print("Some IDs are not assigned, because available for more than one player and unable to decide")
                break
        #end while: so some IDs are added
        
        # TODO: DOING if nothing added (added_ids empty), check the second most common ID in the available_ids (sorted_ids[1])

        # If an ID is present in the same frame as the ID representing a player, it is descarded for that player, since it can't be the same player
        for frame_data in all_data:
            frame_players_ids = [player['id'] for player in frame_data['players']]
            for i, added_id in enumerate(added_ids):
                if added_id in frame_players_ids:
                    available_ids_list[i].difference_update([id for id in frame_players_ids if id != added_id])
        
    return selected_ids_list
    

def get_positions(all_data, selected_ids_list):
    """
    Take selected_ids_list and return the position of the Players in each frame
    all_data: a list of dictionaries (one per frame), each is {'frame_num', 'players'}. 'players' is a list of dictionaries (one per player), each is {'id', 'position'}
    selected_ids_list: a list of 4 sets of IDs, one set for each Player, containing his IDs.
    """
    total_frames = len(all_data)

    player_positions = [[None]* total_frames for _ in range(4)]

    for frame_num, frame_data in enumerate(all_data):
        for player in frame_data['players']:
            for i, selected_ids in enumerate(selected_ids_list):
                if player['id'] in selected_ids:
                    player_positions[i][frame_num] = player['position']
    
    return player_positions

def create_heatmaps(player_positions, output_path = None, alpha=0.05):
    """
    Create heatmaps for each player, given the player_positions list.
    player_positions: a list of 4 lists, each containing the positions of a player.
    output_path: path where to save the images. If None, the images are shown instead of saved.
    alpha: transparency of the points in the heatmap
    """
    
    # Background
    court_img = mpimg.imread('Field.png')
    bg_color = (0.54, 0.73, 0.82)

    for i, positions in enumerate(player_positions):
        if len(positions) == 0:
            print(f"No position data for player {i}")
        else:
            plt.figure(facecolor=bg_color)
            plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image
            plt.scatter(positions[:, 0], positions[:, 1], color=f'C{i+1}', alpha=alpha, label=f"Player {i+1}")
            plt.xticks([])
            plt.yticks([])
            plt.legend()
            if output_path is not None:
                plt.savefig(f"{output_path}player{i+1}.png", bbox_inches='tight')
            else:
                plt.show()
