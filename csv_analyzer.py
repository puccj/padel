import os
import csv
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from tqdm import tqdm
from itertools import combinations

from padel_utils import get_distance

MIN_CO_OCCURRENCE = 5   # Minimum number of co-occurrences to consider two IDs as conflicting


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
        self.selected_ids_list = self._calculate_ids()
        self.players_data = self._calculate_players_data()

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

    def _calculate_ids(self):
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

        # Sort id by frequency (descending order)
        sorted_ids = [id for id, _ in id_counter.most_common()]
        assigned_ids = set()

        # Initialize selected_ids_list with 4 IDs that *needs* to belong to different players.
        # To ensure that, we'll check that the starting ids appear in the same frame 
        # (min_co_occurrence*2 times to be extra sure) and thus can't represent the same player

        # Option A (The easy way): just take the first (valid) 4 in cronological order (i.e the minimum values)
        # starting_ids = sorted(ids_in_field)[:4]

        # Option B (The middle way): the the first (valid) 4 IDs after 5 minutes of video
        # starting_ids = []
        # for frame_data in self.all_data:
        #     if frame_data['frame_num'] < self.fps*60*5:  # 5 minutes
        #         continue
        #     for detection in frame_data['detections']:
        #         if detection['id'] in ids_in_field and detection['id'] not in starting_ids:
        #             starting_ids.append(detection['id'])
        #         if len(starting_ids) == 4:
        #             break
        #     if len(starting_ids) == 4:
        #         break

        # Option C (The better way): starting from the most common IDs, going down to the less common until we find 4 IDs that are present in the same frame
        all_together = False
        id_to_take = 3
        
        while not all_together:
            id_to_take += 1

            # Generate only combinations that include the newest ID
            newest_id = sorted_ids[id_to_take - 1]
            combinations_to_check = list(combinations(sorted_ids[:id_to_take - 1], 3))
            combinations_to_check = [tuple(sorted(list(combo) + [newest_id])) for combo in combinations_to_check]
            
            # Check the combinations to see if any are valid (ids in the same frame)
            for combination in combinations_to_check:
                count = 0
                for frame_data in self.all_data:
                    frame_ids = [detection['id'] for detection in frame_data['detections']]
                    if all(id in frame_ids for id in combination):
                        # Debug
                        total_seconds = frame_data['frame_num'] / self.fps
                        minutes = int(total_seconds // 60)
                        seconds = int(total_seconds % 60)
                        print(f"Debug: all id present together at frame {frame_data['frame_num']} ({minutes:02d}:{seconds:02d}), after checking the {id_to_take}th most common IDs")

                        count += 1
                        if count > MIN_CO_OCCURRENCE * 2:
                            starting_ids = list(combination)
                            all_together = True
                            break
                if all_together:
                    break
        
        print(f"Debug: starting IDs: {starting_ids}")

        # Just for safety. Redundant with Option C
        if not all_together:
            raise NotImplementedError(f"The 4 starting IDs {starting_ids} are not present in a same frame. Unsure if they belong to different players. Aborting...")

        # Remove Assigned IDs for all players
        selected_ids_list = [set([starting_ids[i]]) for i in range(4)]
        for available_ids in available_ids_list:
            available_ids.difference_update(starting_ids)
        assigned_ids.update(starting_ids)

        # TODO: Below a repetition of the following code: may be refactored into a function
        # Remove conflicts from available_ids_list.
        # There is a conflict if an ID is present in the same frame as the ID representing a player.
        # In that case, it is discarded for that player, since it can't represent the same player.
        # TODO: Instead of removing the ID if present together in a single frame, remove it only if present together in N frames

        co_occurrence_counts = [defaultdict(int) for _ in range(len(starting_ids))]

        for frame_data in self.all_data:
            frame_ids = [d['id'] for d in frame_data['detections']]
            for i, selected_id in enumerate(starting_ids):
                if selected_id in frame_ids:
                    # available_ids_list[i].difference_update(frame_ids)  # Remove all IDs present in the same frame as selected_id
                    for id in frame_ids:
                        if id != selected_id:
                            co_occurrence_counts[i][id] += 1
        
        for i in range(len(starting_ids)):
            to_remove = {id for id, count in co_occurrence_counts[i].items() if count > MIN_CO_OCCURRENCE}
            available_ids_list[i].difference_update(to_remove)

        # Now, iteratively assign IDs to players

        # while there are element in at least one of the available_ids_list
        while any(available_ids_list):  # let an iteration of this loop be called a 'round'
            assigned = False    # check if any ID was assigned in this round
            # Traverse IDs in descending order of frequency
            for cand in sorted_ids:
                if cand in assigned_ids:
                    continue

                # Find with players still have this candidate
                possible_players = [i for i in range(4) if cand in available_ids_list[i]]

                if len(possible_players) == 1:  # Assign the ID to the player
                    i = possible_players[0]
                    selected_ids_list[i].add(cand)
                    # available_ids_list[i].remove(cand) will be removed later (in frame_ids)
                    assigned_ids.add(cand)
                    assigned = True

                    # Remove conflicts (co-occurrences) for this player
                    co_occurrence_counts = defaultdict(int)
                    for frame_data in self.all_data:
                        frame_ids = [d['id'] for d in frame_data['detections']]
                        if cand in frame_ids:
                            # available_ids_list[i].difference_update(frame_ids)
                            for id in frame_ids:
                                if id != cand:
                                    co_occurrence_counts[id] += 1

                    to_remove = {id for id, count in co_occurrence_counts.items() if count > MIN_CO_OCCURRENCE}
                    available_ids_list[i].difference_update(to_remove)

                    break  # break the for cand loop and restart from the most common ID

            if not assigned:
                print("Some IDs are not assigned, because available for more than one player and unable to decide")
                print("Unassigned IDs: ", sorted(available_ids_list[0]))    # just the first one is enough, since they are all the same
                # sort unassigned ID by frequency and print their frequency
                unassigned_ids = sorted(available_ids_list[0], key=lambda x: id_counter[x], reverse=True)
                print("Sorted: ", unassigned_ids)

                return selected_ids_list

        return selected_ids_list
    

    def _calculate_players_data(self):
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
    
    def get_selected_ids(self):
        return self.selected_ids_list
    
    def get_players_data(self):
        return self.players_data

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

    def create_mini_court(self, input_path = 'Default', output_path = 'Default'):
        # TODO: Put together the three periods in a single video
        """Draw a post-processed mini-court on top of the original video.

        Parameters
        ----------
        input_path : str, optional
            Path to the input video. Defaults to "input_videos/{video_name}".
        output_path : str, optional
            Path where to save the video. If None, the video will be shown instead of saved.
            Defaults to "to_be_uploaded/{video_name}-postanalyzed/".
        """

        # Remove "-period1" from the video name
        if self.video_name.endswith("-period1"):
            video_name = self.video_name[:-8]
            starting_frame = 0
        if self.video_name.endswith("-period2"):
            video_name = self.video_name[:-8]
            starting_frame = self.fps*60*30
        if self.video_name.endswith("-period3"):
            video_name = self.video_name[:-8]
            starting_frame = self.fps*60*60

        if input_path == 'Default':
            input_path = f"input_videos/{video_name}.mp4"
        
        if output_path == 'Default':
            output_path = f"to_be_uploaded/{video_name}-postanalyzed.mp4"

        # Load the video
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Create a VideoWriter object to save the output video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Fixed variables
        zoom = int(height / 40)
        offset = 2*zoom
        bg_color = (209, 186, 138)
        field_color = (129, 94, 61)
        line_color = (255, 255, 255)
        net_color = (0, 0, 0)
        alpha_field = 0.2
        field_pos = (offset*2, offset*2)

        players_color=[(0,255,255), (0,255,0), (0,0,255), (255,0,0)]
        alpha_player = 0.05

        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num >= len(self.players_data[0]):
                print(f"Frame number {frame_num} exceeds the number of frames in the players data ({len(self.players_data[0])}).")
                break

            # Draw rectangles
            shapes = np.zeros_like(frame,np.uint8)
            cv2.rectangle(shapes, (field_pos[0]-offset, field_pos[1]-offset), (10*zoom+field_pos[0]+offset, 20*zoom+field_pos[1]+offset), bg_color, cv2.FILLED)
            cv2.rectangle(shapes, field_pos, (10*zoom+field_pos[0], 20*zoom+field_pos[1]), field_color, cv2.FILLED)
            # cv2.rectangle(shapes, (field_pos[0]-offset, int(20.25*zoom +field_pos[1]+offset)), (16*zoom+field_pos[0]+offset, 28*zoom+field_pos[1]), (125,125,125), cv2.FILLED)
            out = frame.copy()
            mask = shapes.astype(bool)
            out[mask] = cv2.addWeighted(frame, alpha_field, shapes, 1 - alpha_field, 0)[mask]

            frame = out     # TO SEE: maybe .copy() is needed?

            frame = out     # TO SEE: maybe .copy() is needed?

            # Draw court lines
            cv2.line(frame, (field_pos[0]       ,    3  *zoom+field_pos[1]) , (10*zoom+field_pos[0],      3  *zoom+field_pos[1]) , line_color, 2)  #horizontal
            cv2.line(frame, (field_pos[0]       ,   17  *zoom+field_pos[1]) , (10*zoom+field_pos[0],     17  *zoom+field_pos[1]) , line_color, 2)  #horizontal
            cv2.line(frame, (5*zoom+field_pos[0],int(2.7*zoom+field_pos[1])), ( 5*zoom+field_pos[0], int(17.3*zoom+field_pos[1])), line_color, 2)  #vertical
            cv2.line(frame, (field_pos[0]       ,   10  *zoom+field_pos[1]) , (10*zoom+field_pos[0],     10  *zoom+field_pos[1]) , net_color , 1)  #net

            # Draw players on mini court
            for player_num, player_data in enumerate(self.players_data):
                player_info = player_data[frame_num]
                if player_info['position'] is None:
                    continue

                # Draw the player position
                cv2.circle(frame, (int(player_info['position'][0]*zoom+field_pos[0]),int(player_info['position'][1]*zoom+field_pos[1])), 1, players_color[player_num], 3, cv2.LINE_AA)

                # Draw the speed and distance below the court
                # text = f"Player {player_num+1}: {int(player_info['distance'])} m, {player_info['speed']:.2f} km/h"
                # cv2.putText(frame, text, (field_pos[0]-offset, int(20*zoom+field_pos[1]+offset + (player_num+1)*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, players_color[player_num], 1, cv2.LINE_AA)

            if output_path is None:
                cv2.imshow("Mini Court", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                out_video.write(frame)

            frame_num += 1

        # Release the video objects
        cap.release()
        if output_path is None:
            cv2.destroyAllWindows()
        else:
            out_video.release()


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