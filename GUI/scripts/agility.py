import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from scripts.utils import *
from scripts.segment import Segmenter
from scripts.track import Tracker

def agility_measure(PATH: str) -> None:

  CSV_SAVE_NAME = os.path.basename(PATH).split('.')[0]+'_stats.csv'
  CSV_SAVE_PATH = os.path.join(os.path.join(os.path.abspath('.'), 'data'), CSV_SAVE_NAME)
  PLAYERS_SAVE_NAME = os.path.basename(PATH).split('.')[0]+'_players'
  PLAYERS_SAVE_PATH = os.path.join(os.path.join(os.path.abspath('.'), 'data'), PLAYERS_SAVE_NAME)

  cap = cv2.VideoCapture(PATH)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  # Video's lines (user input)
  start_lines_map = {'Agility_trimmed.mp4': ([1401, 706], [1223, 516]),
                    'agility.mp4': ([1081, 675], [1074, 599]),
                    'Agility_t.MP4': ([346, 843], [546, 737])}


  end_lines_map = {'Agility_trimmed.mp4': ([1401, 706], [1223, 516]),
                  'agility.mp4': ([1081, 675], [1074, 599]),
                  'Agility_t.MP4': ([346, 843], [546, 737])}


  start_line = start_lines_map[os.path.basename(PATH)]
  end_line = end_lines_map[os.path.basename(PATH)]

  # Flag that determines video needs to be flipped or not
  flip = False  # ---> video has left orientation

  # Get center point of the start line
  start_line_center_x = (start_line[0][0] + start_line[1][0])//2
  start_line_center_y = (start_line[0][1] + start_line[1][1])//2
  start_line_center = (start_line_center_x, start_line_center_y)

  # Get points with respect to left orientation
  if start_line_center[0] > width/2:

    flip = True  # video has right orientation, needs to be flipped

    # Get start and end lines with respect to left orientation
    start_line[0][0] = width - start_line[0][0]
    start_line[1][0] = width - start_line[1][0]

    # Get meter line with respect to left orientation
    end_line[0][0] = width - end_line[0][0]
    end_line[1][0] = width - end_line[1][0]


  # Get center point of the start and lines
  start_line_center_x = (start_line[0][0] + start_line[1][0])//2
  start_line_center_y = (start_line[0][1] + start_line[1][1])//2
  start_line_center = (start_line_center_x, start_line_center_y)

  end_line_center_x = (end_line[0][0] + end_line[1][0])//2
  end_line_center_y = (end_line[0][1] + end_line[1][1])//2
  end_line_center = (end_line_center_x, end_line_center_y)

  # detector = Detector()
  tracker = Tracker(distance_threshold=np.inf, not_exist_threshold=np.inf)
  segmenter = Segmenter()

  # Initialize opacity for blending
  opacity = 0.2

  # Initialize variables used below
  last_player_id = None
  last_masked_frame = None
  start = False
  frame_count = 0
  players_stats = {} # Save players's stats

  # Loop through each frame of the video and apply the segmentation code
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  for i in tqdm(range(total_frames)):

    ret, frame = cap.read()

    if not ret:
      break

    yield int(i/total_frames * 100) 

    if flip:
      frame = cv2.flip(frame, 1)

    if not start:

      ################# Draw last player's stats #################
      if last_player_id is not None:
        frame = draw_agility(frame, flip, last_player_id, frame_count)

      ################# initial mask at start of the video or player just finished his test #################
      if last_masked_frame is None:

        # Apply the mask to the original image
        masked_frame = default_mask(frame, start_line)

      ################## Apply segmentation on masked frame #################
      segmented_mask, _, _ = segmenter.segment(masked_frame)

      ################# Get bboxes of objects #################
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_mask)

      ############## Found background only (NO PLAYERS EXIST) #################

      """ NOTE: ** Validate on 2nd frame ** """
      if len(stats) == 1:
        masked_frame = last_masked_frame
        continue
      else:
        last_masked_frame = masked_frame

      ############## Get player boundary box #################
      largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

      player_centroid = centroids[largest_label]

      player_stats = stats[largest_label]

      xmin, ymin, w, h, area = player_stats
      xmax, ymax = xmin+w, ymin+h
      player_stats = [[xmin, ymin, xmax, ymax]]

      ############## Limit player boundary box posistion with respect to start line #################
      if start_line_center[0] - player_centroid[0] > 150:
        last_masked_frame = None
        continue

      ############## Get player's forefoot #################

      player_forefoot = get_player_forefoot(segmented_mask)

      ############## No player exists to start testing #################
      if not player_forefoot:
        last_masked_frame = None
        continue

      ############## OUTLIERS (OUT OF TESTING AREA)  #################
              ######## NOTE: You can try meter line instead of start line ########

      if player_forefoot[1] > start_line[0][1]: ######## (ITS y > y of the start line) ########
        last_masked_frame = None
        continue

      ############## Update masked frame to fit player's boundary box #################
      # Apply the mask to the original image
      masked_frame = update_mask(frame, player_stats[0])

      ############## Player's forefoot passed start line ##############
      if player_forefoot[0] - max(start_line[0][0], start_line[1][0]) > 50:
        last_masked_frame = None
        frame_count = 0
        start=True

      ########################## Save player's image & draw player's rectangle and ID ##########################
      else:

        # Track players
        tracked_players, disappeared_players_count, new_players, distances = tracker.track(player_stats)

        # Useful for rectangles transparency
        overlay = frame.copy()

        # Loop through tracked players
        for player_id, player_bbox in tracked_players.items():

          # Convert (xmin, ymin, xmax, ymax) as integers
          xmin, ymin, xmax, ymax = player_bbox = [int(x) for x in player_bbox]

          #################### Save player image ####################
          save_image(frame, flip, i, PLAYERS_SAVE_PATH, player_id, player_bbox)

          #################### Draw player's rectangle and ID ####################

          frame = draw_player_bbox(frame, flip, player_id, player_bbox)

        #################### Blend overlay with original image using alpha channel ####################
        frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)


    ############## started testing #################
    if start:

      ############## Apply segmentation on masked frame ##############
      segmented_mask, _, _ = segmenter.segment(masked_frame)

      ############## Get bboxes of objects ##############
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_mask)

      ############## Found background only #################

      """ NOTE: ** Validate on 2nd frame ** """
      if len(stats) == 1:
        masked_frame = last_masked_frame
        continue
      else:
        last_masked_frame = masked_frame


      ############## Get player boundary box #################
      largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

      player_centroid = centroids[largest_label]

      player_stats = stats[largest_label]

      xmin, ymin, w, h, area = player_stats
      xmax, ymax = xmin+w, ymin+h
      player_stats = [[xmin, ymin, xmax, ymax]]

      ############## Update masked frame to fit player's boundary box #################
      # Apply the mask to the original image
      masked_frame = update_mask(frame, player_stats[0])

      ########################## Save player's image & draw player's rectangle and ID ##########################

      # Track players
      tracked_players, disappeared_players_count, new_players, distances = tracker.track(player_stats)

      # Useful for rectangles transparency
      overlay = frame.copy()

      # Loop through tracked players
      for player_id, player_bbox in tracked_players.items():

        # Convert (xmin, ymin, xmax, ymax) as integers
        xmin, ymin, xmax, ymax = player_bbox = [int(x) for x in player_bbox]

        #################### Save player image ####################
        save_image(frame, flip, i, PLAYERS_SAVE_PATH, player_id, player_bbox)

        #################### Draw player's rectangle and ID ####################

        frame = draw_player_bbox(frame, flip, player_id, player_bbox)

      #################### Get player's forefoot ####################
      player_forefoot = get_player_forefoot(segmented_mask)

      #################### Blend overlay with original image using alpha channel ####################
      frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

      #################### Distance between player's forefoot and end line ####################

      # Get distance between player's forefoot and end line
      distance = player_forefoot[0] - min(end_line[0][0], end_line[1][0])

      ########################## still running ##########################
      if distance > -100:
        frame_count += 1
        frame = draw_agility(frame, flip, player_id, frame_count)

        continue

      ########################## Finished Test ##########################

      # Save player's stats (Seconds needed to complete test)
      players_stats[player_id] = frame_count/fps

      # reset settings for the following player test
      last_masked_frame = None
      start = False
      last_player_id = player_id

      # Drop (reset) tracker
      tracker.reset()

  ########################## Save CSV file ##########################

  # Dictionary to dataframe
  players_stats_df = pd.DataFrame.from_dict(players_stats, orient='index', columns=['Time (min)'])

  # Add a 'Player ID' column with the index values
  players_stats_df.insert(0, 'Player ID', players_stats_df.index)

  # Save statistics dataframe in CSV file
  players_stats_df.to_csv(CSV_SAVE_PATH, index=False)

  cap.release()

if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Measure player's agility")
    parser.add_argument('PATH', help='Path of the video to be processed')

    # Parse arguments
    args = parser.parse_args()

    # Move annotated images
    agility_measure(args.PATH)
