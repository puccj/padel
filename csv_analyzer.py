import os
import csv
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from tqdm import tqdm

from padel_utils import get_distance

def is_inside_field(position):
    return 0 <= position[0] <= 10 and 0 <= position[1] <= 20

class CsvAnalyzer:
    def __init__(self, input_csv_path, fps=None, mean_interval=None):
        """
        Initialize the CsvAnalyzer object.
        
        Parameters
        ----------
        input_csv_path : str
            Path to the csv file to analyze.
        fps : int
            Optional number of frames per second of the video. If not provided, it will be read from the <cam_name>-fps.txt file.
        mean_interval : int
            Optional number of frames to consider for the mean position and velocity (default: 1*fps = 1 second).
        """

        self.video_name = os.path.basename(input_csv_path)
        self.video_name = os.path.splitext(self.video_name)[0]

        if fps is None:
            self._read_fps()
        else:
            self.fps = fps

        self.mean_interval = int(mean_interval or 1*self.fps)    # Number of frames to consider for the mean position and velocity (2*fps = 2 seconds)

        self.all_data = self._read_csv(input_csv_path)
        self.selected_ids_list = self._get_ids()
        self.players_data = self._get_players_data()

    def _read_fps(self):
        """Read the fps from the <cam_name>-fps.txt file."""

        fps_files = [file for file in os.listdir(os.getcwd()) if file.endswith("-fps.txt")]
        if len(fps_files) > 1:
            print(f"Warning: More than one fps file found. Using the first one ({fps_files[0]})."
                   " Delete the others if not needed or manually set fps in CSVAnalyzer.")
        if fps_files:
            with open(os.path.join(os.getcwd(), fps_files[0]), 'r') as f:
                self.fps = float(f.read().strip())
        else:
            raise FileNotFoundError("FPS file not found. Please provide the fps manually.")

    def _read_csv(self, input_csv_path):
        """Reads a CSV file and organizes its data into a structured format.

        Parameters
        ----------
        input_csv_path : str
            The path to the input CSV file.

        Returns
        -------
        organized_data: list
            A list of dictionaries, each containing the frame number and a list of detections. 
            Each detection is represented as a dictionary with 'id' and 'position' keys.

        Raises
        ------
        FileNotFoundError
            If the specified CSV file does not exist.
        ValueError
            If the specified CSV file is empty.

        Notes
        -----
        - The CSV file is expected to have rows where the first column is the frame number (integer),
            followed by pairs of detection ID (integer) and detection position (string in the format '[x y]').
        - If a detection ID is not an integer, an error message will be printed indicating the frame number.
        """

        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"File not found: {input_csv_path}")
        
        if os.path.getsize(input_csv_path) == 0:
            raise ValueError(f"File is empty: {input_csv_path}")

        organized_data = []

        with open(input_csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                frame_num = int(row[0])
                detections_data = row[1:]
                
                detections = []
                for i in range(0, len(detections_data), 2):
                    try:
                        detection_id = int(detections_data[i])
                    except ValueError:
                        print(f"Error in frame {frame_num}: detection ID is not an integer. Is there a trailing comma?.")
                    detection_position_str = detections_data[i + 1]
                    detection_position_str = detection_position_str.strip('[] ')                      # Remove brackets and spaces
                    detection_position = [float(coord) for coord in detection_position_str.split()]   # Convert to list of floats
                    detections.append({'id': detection_id, 'position': detection_position})
                
                frame_data = {'frame_num': frame_num, 'detections': detections}
                organized_data.append(frame_data)
        
        return organized_data

    def _get_ids(self):
        """Analyzes the organized data and returns the IDs of the 4 players throughout the video.

        Returns
        -------
        selected_ids_list: list
            A list of 4 sets, each containing the IDs of a player.
        """

        # Remove IDs that are never inside the field
        ids_in_field = set()

        for frame_data in self.all_data:
            for detection in frame_data['detections']:
                if detection['id'] in ids_in_field:
                    continue
                if is_inside_field(detection['position']):
                    ids_in_field.add(detection['id'])

        available_ids_list = [ids_in_field.copy() for _ in range(4)] # 4 copies of all the IDs that are inside the field

        # Count the number of times each (valid) ID appears
        id_counter = Counter()
        for frame_data in self.all_data:
            for detection in frame_data['detections']:
                if detection['id'] in ids_in_field:
                    id_counter[detection['id']] += 1


        # Initialize selected_ids_list with 4 IDs that *needs* to belong to different players

        # Option A (The easy way): just take the first (valid) 4 in cronological order (i.e the minimum values)
        starting_ids = sorted(ids_in_field)[:4]

        # Option B (The better way): starting from the most common IDs, going down to the less common untill we find 4 IDs that are present in the same frame
        # # Get the 4 most common IDs and use to initialize the selected_ids sets
        # # TODO: Warning: it could happen that two of the most common IDs are the same player.
        # #       For now I'll just check if all 4 most common IDs are simultaneously present at least in one frame 
        # #       (but do nothing if that is the case)
        # #       Maybe I just need to check the 5th most common ID, excluding one of the previous 4 (trying with all combination) 
        # #       and check again if these new 4 IDs are simultaneous in one frame.  And so on cheching the 6th, 7th, etc. most common IDs

        # most_common_ids = [id for id, _ in id_counter.most_common(4)]
        # starting_ids = most_common_ids

        # Check if the 4 starting IDs are present in the same frame
        all_together = False
        for frame_data in self.all_data:
            frame_ids = [detection['id'] for detection in frame_data['detections']]
            if all(id in frame_ids for id in starting_ids):
                all_together = True
                break
        if not all_together:
            raise NotImplementedError("The 4 starting IDs are not present in a same frame. Unsure if they belong to different players. Aborting...")
        
        selected_ids_list = [set([starting_ids[i]]) for i in range(4)]
        for available_ids in available_ids_list:
            available_ids.difference_update(starting_ids)  # Assigned IDs are not available for anyone

        #TODO: Make this a function
        # If an ID is present in the same frame as the ID representing a player, it is descarded for that player, since it can't be the same player
            # for frame_data in all_data:
            #     frame_players_ids = [player['id'] for player in frame_data['players']]
            #     for i, selected_ids in enumerate(selected_ids_list):
            #         for selected_id in selected_ids:
            #             if selected_id in frame_players_ids:
            #                 available_ids_list[i].difference_update([id for id in frame_players_ids if id != selected_id])
        for frame_data in self.all_data:
            frame_detections_ids = [detection['id'] for detection in frame_data['detections']]
            for i, selected_id in enumerate(starting_ids):
                if selected_id in frame_detections_ids:
                    # TODO: check which of the following is faster:
                    # available_ids_list[i].difference_update([id for id in frame_detections_ids if id != selected_id])
                    available_ids_list[i].difference_update([id for id in frame_detections_ids])

        # Check if there are still available IDs for each player. If so, add them to the selected_ids (and remove from available)
        
        #while there are element in at least one of the available_ids_list
        while any(available_ids_list):
            multiple_ids = set()    # will contain the IDs present in more than one available_ids list
            added_ids = [None]*4    # will contain the ID added for each player
            detection_to_take = -1     # index of the detection that will be checked

            while not any(added_ids):
                one_is_long_enough = False
                detection_to_take += 1

                for i, available_ids in enumerate(available_ids_list):    #for each Player
                    if len(available_ids) <= detection_to_take:
                        # TODO: remove this debug print
                        print(f"Player {i} has not enough available IDs to check the {detection_to_take}th most common ID")
                        continue
                    one_is_long_enough = True
                    sorted_id = sorted(available_ids, key = lambda id: id_counter[id], reverse=True)[detection_to_take]
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
                    print("Unassigned IDs: ", available_ids_list[0])    # just the first one is enough, since they are all the same
                    return selected_ids_list
            #end while: so some IDs are added

            # If an ID is present in the same frame as the ID representing a player, it is descarded for that player, since it can't be the same player
            for frame_data in self.all_data:
                frame_detections_ids = [detection['id'] for detection in frame_data['detections']]
                for i, added_id in enumerate(added_ids):
                    if added_id in frame_detections_ids:
                        available_ids_list[i].difference_update([id for id in frame_detections_ids if id != added_id])
            
        return selected_ids_list
    

    def _get_players_data(self):
        """Take selected_ids_list and return the data of the Players in each frame
        
        Returns
        -------
        players_data: list
            A list of 4 lists (one per Players), each containing a dictionary for each frame.
            Each dictionary contains the 'position', 'distance' and 'speed' of the player in that frame.
        """

        total_frames = len(self.all_data)

        players_data = [[{'position': None, 'distance': 0, 'speed': 0} for _ in range(total_frames)] for _ in range(4)]

        for frame_num, frame_data in enumerate(self.all_data):
            for player, selected_ids in enumerate(self.selected_ids_list):
                player_found = False

                for detection in frame_data['detections']:
                    
                    if detection['id'] not in selected_ids:
                        continue

                    player_found = True

                    if detection['position'] is None:     #TODO: remove this check (or solve if raised)
                        raise NotImplementedError("WHAT?")
                    
                    position = detection['position']
                    players_data[player][frame_num]['position'] = position

                    if frame_num <= self.mean_interval:
                        continue

                    # data of n=mean_interval frames ago
                    old_data = players_data[player][frame_num - self.mean_interval]

                    if old_data['position'] is None:
                        players_data[player][frame_num]['distance'] = players_data[player][frame_num - 1]['distance']
                        continue

                    distance = get_distance(position, old_data['position'])

                    # add distance only between each n frame, oterwise I'll add to much (non ho voglia di spiegare)
                    if frame_num % self.mean_interval == 0:
                        cumulative_distance = old_data['distance'] + distance
                    else:
                        cumulative_distance = players_data[player][frame_num - 1]['distance']
                    
                    players_data[player][frame_num]['distance'] = cumulative_distance

                    dt = self.mean_interval / self.fps
                    players_data[player][frame_num]['speed'] = 3.6* distance / dt

                    break   # break detections loop (if here, the player has been found, so no need to check other detections)
                
                if not player_found:
                    # players_data[player][frame_num]['position'] = None    
                    players_data[player][frame_num]['distance'] = players_data[player][frame_num - 1]['distance']
                    # print(f"Debug: Player {player} not found in frame {frame_num}")

            # end for player
        # end for frame_data

        return players_data

    def create_heatmaps(self, output_path = 'Default', alpha=0.05, colors=['yellow', (0,1,0), (1,0,0), 'blue'], draw_net=False):
        """Create heatmaps for each player, containing also distance and speed data.
        
        Parameters
        ----------
        output_path : str, optional
            Path where to save the heatmaps. If None, the heatmaps will be shown instead of saved. 
            Defauts to "to_be_uploaded/{video_name}-heatmaps/".
        alpha : float, optional
            Transparency of the player positions. Defaults to 0.05.
        colors : list, optional
            List of colors to use for each player. Defaults to yellow, green, red, blue.

        Notes
        -----
        The heatmaps will be saved as "player{i}.png" for each player and "all_players.png" for all players.
        """

        if output_path == 'Default':
            output_path = f"to_be_uploaded/{self.video_name}-heatmaps/"
        
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Background
        if draw_net:
            court_img = mpimg.imread('Field with net.png')
        else:
            court_img = mpimg.imread('Field.png')
        bg_color = (0.54, 0.73, 0.82)

        for i, player_data in enumerate(self.players_data):    # for each player
            positions = [frame['position'] for frame in player_data]

            if len(positions) == 0:
                print(f"No position data for player {i}")
                continue

            #TODO: Caclulate total distance like below.
            total_distance = 0

            first_not_None = 0            
            while positions[first_not_None] is None:
                print("First position is None")
                first_not_None += 1

            X = [positions[first_not_None][0]]
            Y = [positions[first_not_None][1]]

            for j in range(1, len(positions)):
                if positions[j] is not None:
                    X.append(positions[j][0])
                    Y.append(positions[j][1])
                    if positions[j-1] is not None:
                        total_distance += get_distance(positions[j], positions[j-1])

            # distances = [get_distance(positions[j], positions[j-1]) for j in range(1, len(positions)) if positions[j] is not None and positions[j-1] is not None]
            # np.sum(distances)
            # X = [pos[0] for pos in positions if pos is not None]
            # Y = [pos[1] for pos in positions if pos is not None]


            plt.figure(facecolor=bg_color)
            plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image
            plt.scatter(X, Y, color=colors[i], alpha=alpha, label=f"Player {i+1}")
            plt.xticks([])
            plt.yticks([])
            leg = plt.legend()
            for lh in leg.legend_handles:
                lh.set_alpha(1)
            plt.gca().invert_yaxis()
            text = f"Total distance: {int(total_distance)} m\n"
            text += f"Mean velocity: {np.mean([frame['speed'] for frame in player_data]):.2f} km/h"

            bbox_props = dict(boxstyle="round,pad=0.5", edgecolor="grey", facecolor="white", alpha=0.7)
            plt.text(0.5, 21.5, text, fontsize=12, verticalalignment='center', bbox=bbox_props)

            if output_path is not None:
                plt.savefig(f"{output_path}player{i+1}.png", bbox_inches='tight')
            else:
                plt.show()
        
        
        plt.figure(facecolor=bg_color)
        text = f"Total distances:  |  Mean velocities\n\n"
        for i, player_data in enumerate(self.players_data):
            text += f"Player {i+1}: {int(player_data[-1]['distance'])} m        "
            text += f"{np.mean([frame['speed'] for frame in player_data]):.2f} km/h\n"

            positions = [frame['position'] for frame in player_data]
            if len(positions) == 0:
                continue
            X = [pos[0] for pos in positions if pos is not None]
            Y = [pos[1] for pos in positions if pos is not None]
            plt.imshow(court_img, extent=[0, 10, 0, 20])
            # plt.imshow(court_img, extent=[0, 10, 0, 20])  # Extent sets the x and y limits of the image
            plt.scatter(X, Y, color=colors[i], alpha=alpha, label=f"Player {i+1}")
            plt.xticks([])
            plt.yticks([])

        leg = plt.legend(loc=(1.05, 0))        
        # Set legend points to be fully opaque
        for lh in leg.legend_handles:
            lh.set_alpha(1)
        plt.gca().invert_yaxis()
        plt.text(11, 5, text, fontsize=12, verticalalignment='center', bbox=bbox_props)

        if output_path is not None:
            plt.savefig(f"{output_path}all_players.png", bbox_inches='tight')
        else:
            plt.show()


    def create_speed_graph(self, output_path = 'Default', figsize=(8, 6)):
        """Create a graph showing the speed of each player over time.

        Parameters
        ----------
        output_path : str, optional
            Path where to save the graph. If None, the graph will be shown instead of saved.
            Defaults to "to_be_uploaded/{video_name}-graphs/speed_graph.png".
        figsize : tuple, optional
            Size of the figure. Defaults to (8, 6).
        """

        if output_path == 'Default':
            output_path = f"to_be_uploaded/{self.video_name}-graphs/speed_graph.png"

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=figsize)

        for i, player_data in enumerate(self.players_data):
            speeds = [frame['speed'] for frame in player_data]
            plt.plot(speeds, label=f"Player {i+1}")

        plt.xlabel("Frame number")
        plt.ylabel("Speed (km/h)")
        plt.legend()

        if output_path is not None:
            plt.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()
    
    def create_distance_graph(self, output_path = 'Default', figsize=(8, 6)):
        # TODO: calculate the distance correctly
        """Create a graph showing the distance covered by each player over time.

        Parameters
        ----------
        output_path : str, optional
            Path where to save the graph. If None, the graph will be shown instead of saved.
            Defaults to "to_be_uploaded/{video_name}-graphs/distance_graph.png".
        figsize : tuple, optional
            Size of the figure. Defaults to (8, 6).

        Notes
        -----
        The distance is (erroneously) calculated as the sum of the distances between each pair of consecutive positions.
        """

        if output_path == 'Default':
            output_path = f"to_be_uploaded/{self.video_name}-graphs/distance_graph.png"

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=figsize)

        for i, player_data in enumerate(self.players_data):
            distances = [frame['distance'] for frame in player_data]
            plt.plot(distances, label=f"Player {i+1}")

        plt.xlabel("Frame number")
        plt.ylabel("Distance (m)")
        plt.legend()

        if output_path is not None:
            plt.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()


    def create_videos(self, output_path = 'Default', field_height=800, draw_net=False, speed_factor=2, trace=0, alpha=0.05):
        #TODO See output_path when None...
        """Create a video showing the heatmap of the players over time.

        Parameters
        ----------
        output_path : str, optional
            Path where to save the video. If None, the video will be shown instead of saved.
            Defaults to "to_be_uploaded/{video_name}-heatmaps/".
        field_height : int, optional
            Height of the field in pixels. Defaults to 800.
        draw_net : bool, optional
            Whether to draw the net on the field. Defaults to False.
        speed_factor : int, optional
            Factor by which to speed up the video. Defaults to 2.
        trace : int, optional
            Number of frames after which the trace fades away
        alpha : float, optional
            Transparency of the trace. Defaults to 0.05.

        Notes
        -----
        The video will be saved as "player{i}.mp4" for each player and "all_players.mp4" for all players.
        """

        if output_path == 'Default':
            output_path = f"to_be_uploaded/{self.video_name}-heatmaps/"

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        bg_color = (209, 186, 138)
        players_color=[(0,255,255), (0,255,0), (0,0,255), (255,0,0)]
        alpha_box = 0.7
        
        offset = int(field_height/30)
        font_size = int(field_height/800)
        font_thickness = int(field_height/400)
        field_width = int(field_height/2)
        width_single = field_width + 2*offset
        height_single = field_height + 10*offset
        width_all = int(field_height*1.5 + 2*offset)
        height_all = int(field_height + 2*offset)
        text_yposition = [5*offset, 7*offset, 9*offset, 11*offset]
        
        # Place the field image in the background
        bg_single = np.full((height_single, width_single, 3), bg_color, dtype=np.uint8)
        bg_all = np.full((height_all, width_all, 3), bg_color, dtype=np.uint8)
        if draw_net:
            court_img = cv2.imread('Field with net.png')
        else:
            court_img = cv2.imread('Field.png')
        court_img = cv2.resize(court_img, (field_width, field_height))
        bg_single[offset:offset+field_height, offset:offset+field_width] = court_img
        bg_all[offset:offset+field_height, offset:offset+field_width] = court_img

        # Add rectangle for text
        overlay = bg_single.copy()
        cv2.rectangle(overlay, (offset, field_height + 2*offset), (width_single - offset, height_single - offset), (128, 128, 128), -1)
        cv2.addWeighted(overlay, alpha_box, bg_single, 1 - alpha_box, 0, bg_single)
        overlay = bg_all.copy()
        cv2.rectangle(overlay, (field_width + 3*offset, offset), (width_all - offset, 13*offset), (128, 128, 128), -1)
        cv2.addWeighted(overlay, alpha_box, bg_all, 1 - alpha_box, 0, bg_all)

        # Add fixed text
        text = '      Total distance   |   Speed'
        bg_all = cv2.putText(bg_all, text, (field_width + 7*offset, 3*offset), cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 0), font_thickness, cv2.LINE_AA)
        text = 'Player 1:'
        bg_all = cv2.putText(bg_all, text, (field_width + 4*offset, 5*offset), cv2.FONT_HERSHEY_COMPLEX, font_size, players_color[0], font_thickness, cv2.LINE_AA)
        text = 'Player 2:'
        bg_all = cv2.putText(bg_all, text, (field_width + 4*offset, 7*offset), cv2.FONT_HERSHEY_COMPLEX, font_size, players_color[1], font_thickness, cv2.LINE_AA)
        text = 'Player 3:'
        bg_all = cv2.putText(bg_all, text, (field_width + 4*offset, 9*offset), cv2.FONT_HERSHEY_COMPLEX, font_size, players_color[2], font_thickness, cv2.LINE_AA)
        text = 'Player 4:'
        bg_all = cv2.putText(bg_all, text, (field_width + 4*offset, 11*offset), cv2.FONT_HERSHEY_COMPLEX, font_size, players_color[3], font_thickness, cv2.LINE_AA)

        # Prepare video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_singles = [cv2.VideoWriter(f'{output_path}player{i+1}.mp4', fourcc, self.fps*speed_factor, (width_single, height_single)) for i in range(4)]
        out_all = cv2.VideoWriter(f'{output_path}all_players.mp4', fourcc, self.fps*speed_factor, (width_all, height_all))

        # Initialize past positions storage for trace effect
        past_positions = [[] for _ in range(4)]

        # Write frames
        for frame_num in tqdm(range(len(self.players_data[0]))):
            frame_all = bg_all.copy()

            for player, player_data in enumerate(self.players_data):
                frame_single = bg_single.copy()
                pos = player_data[frame_num]['position']
                if pos is None:
                    out_singles[player].write(frame_single)
                    continue
                
                # Update past positions
                past_positions[player].append(pos)
                if len(past_positions[player]) > trace and trace > 0:
                    past_positions[player].pop(0)
                
                # Draw current position and text
                cv2.circle(frame_single, (int(pos[0]*field_width/10 +offset), int(pos[1]*field_width/10 +offset)), 1, players_color[player], int(field_width/50), cv2.LINE_AA)
                text = f'Player {player+1}'
                frame_single = cv2.putText(frame_single, text, (2*offset, field_height + 4*offset), cv2.FONT_HERSHEY_COMPLEX, font_size, players_color[player], font_thickness, cv2.LINE_AA)
                text = f'Total distance:  {int(player_data[frame_num]['distance'])} m'
                frame_single = cv2.putText(frame_single, text, (2*offset, field_height + 6*offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness, cv2.LINE_AA)
                text = f'Speed:    {player_data[frame_num]['speed']:.2f} km/h'
                frame_single = cv2.putText(frame_single, text, (2*offset, field_height + 8*offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness, cv2.LINE_AA)

                # Draw current position on all players video
                cv2.circle(frame_all, (int(pos[0]*field_width/10 +offset), int(pos[1]*field_width/10 +offset)), 1, players_color[player], int(field_width/50), cv2.LINE_AA)
                text = f'                {int(player_data[frame_num]['distance'])} m        {player_data[frame_num]['speed']:.2f} km/h'
                frame_all = cv2.putText(frame_all, text, (field_width + 4*offset, text_yposition[player]), cv2.FONT_HERSHEY_SIMPLEX, font_size, players_color[player], font_thickness, cv2.LINE_AA)

                # Draw trace positions
                overlay_single = frame_single.copy()
                overlay_all = frame_all.copy()
                for past_pos in past_positions[player]:
                    # alpha_trace = 1.0 - (trace_index + 1) / trace if trace > 0 else 1.0
                    cv2.circle(overlay_single, (int(past_pos[0]*field_width/10 +offset), int(past_pos[1]*field_width/10 +offset)), 1, players_color[player], int(field_width/50), cv2.LINE_AA)
                    cv2.circle(overlay_all, (int(past_pos[0]*field_width/10 +offset), int(past_pos[1]*field_width/10 +offset)), 1, players_color[player], int(field_width/50), cv2.LINE_AA)

                cv2.addWeighted(overlay_single, alpha, frame_single, 1 - alpha, 0, frame_single)
                cv2.addWeighted(overlay_all, alpha, frame_all, 1 - alpha, 0, frame_all)
                
                out_singles[player].write(frame_single)
            out_all.write(frame_all)
        
        out_singles[0].release()
        out_singles[1].release()
        out_singles[2].release()
        out_singles[3].release()
        out_all.release()

    def create_graphs(self, output_path = 'Default', figsize=(8, 6)):
        """Create both the speed and the distance graphs showing the speed and distance of each player over time.
        
        Parameters
        ----------
        output_path : str, optional
            Path where to save the graphs. If None, the graphs will be shown instead of saved.
            Defaults to "to_be_uploaded/{video_name}-graphs/".
        figsize : tuple, optional
            Size of the figure. Defaults to (8, 6).
        """

        self.create_speed_graph(output_path, figsize)
        self.create_distance_graph(output_path, figsize)
    
    def create_heatmaps_and_video(self, output_path = 'Default', 
                                        alpha = 0.05, 
                                        colors = ['yellow', (0,1,0), (1,0,0), 'blue'], 
                                        draw_net = False, 
                                        field_height = 800, 
                                        speed_factor = 2, 
                                        trace = 0):
        """Create both video and final image of the heatmap for each player, containing also distance and speed data.

        Parameters
        ----------
        output_path : str, optional
            Path where to save the heatmaps and the video. If None, the heatmaps and the video will be shown instead of saved.
            Defaults to "to_be_uploaded/{video_name}-heatmaps/".
        alpha : float, optional
            Transparency of the player positions. Defaults to 0.05.
        colors : list, optional
            List of colors to use for each player. Defaults to yellow, green, red, blue.
        draw_net : bool, optional
            Whether to draw the net on the field. Defaults to False.
        field_height : int, optional
            Height of the field in pixels. Defaults to 800.
        speed_factor : int, optional
            Factor by which to speed up the video. Defaults to 2.
        trace : int, optional
            Number of frames after which the trace fades away
        """

        self.create_heatmaps(output_path, alpha, colors, draw_net)
        self.create_videos(output_path, field_height, draw_net, speed_factor, trace, alpha)
    
    def create_all(self, output_path = 'Default', figsize=(8, 6), alpha=0.05, colors=['yellow', (0,1,0), (1,0,0), 'blue'], draw_net=False):
        self.create_heatmaps_and_video(output_path, alpha, colors, draw_net)
        self.create_graphs(output_path, figsize)