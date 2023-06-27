import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Utility function to draw player's distance
def draw_power(frame, flip, player_id, total_distance_px, total_distance_m):
  """
  Draws the player distance in pixels and meters on a given image frame. 
  
   Parameters: 
      frame: numpy array 
         The input image frame on which the player distance is to be drawn.
      
      flip: bool 
         A boolean value indicating whether the frame is to be flipped horizontally or not.

      player_id: int
         An integer value representing the id of the player whose distance is to be drawn.
      
      total_distance_px: int
         An integer value representing the total distance traveled by the player in pixels.
      
      total_distance_m: float
         A float value representing the total distance traveled by the player in meters. 
      
   
   Returns: numpy array 
      A modified image frame with the player distance drawn on it.
  """
  
  # Get dimension with repect to right orientation
  if flip:
     frame = cv2.flip(frame, 1)

  px_text = f'Player {player_id} distance (PX): {total_distance_px}'
  m_text = f'Player {player_id} distance (M): {total_distance_m:.2f}'

  px_org = (20, 72)
  m_org = (20, 140)
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  color = (0, 0, 255)
  thickness = 2
  cv2.rectangle(frame, (0, 0), (500, 200), [255, 255, 255], -1)
  cv2.putText(frame, px_text, px_org, font, font_scale, color, thickness)
  cv2.putText(frame, m_text, m_org, font, font_scale, color, thickness)
  
  # Flip image to save as its original orientation
  if flip:
     frame = cv2.flip(frame, 1)
    
  return frame

# Utility function to draw player's distance
def draw_speed(frame, flip, player_id, frame_count, distance=30, fps=30):
  """
   Draws the player speed in meters per second and time in seconds on a given image frame.

   Parameters:
      frame: numpy array
         The input image frame on which the player speed and time is to be drawn.

      flip: bool
         A boolean value indicating whether the frame is to be flipped horizontally or not.
      
      player_id: int
         An integer value representing the id of the player whose speed is to be drawn.
      
      frame_count: int
         An integer value representing the frame count of the video.
      
      distance: float, optional
         A float value representing the distance covered by the player in meters. Default value is 30 meters.
      
      fps: float, optional
         A float value representing the frames per second of the video. Default value is 30 fps.

   Returns: numpy array
      A modified image frame with the player speed and time drawn on it.
  """
  
  # Get dimension with repect to right orientation
  if flip:
     frame = cv2.flip(frame, 1)

  frame_text = f'Player {player_id} time: {frame_count/fps:.1f} SEC'
  seconds_text = f'Player {player_id} distance: {30/(frame_count/fps):.2f} M/S'

  frame_org = (20, 72)
  seconds_org = (20, 140)
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  color = (0, 0, 255)
  thickness = 2
  
  cv2.rectangle(frame, (0, 0), (500, 200), [255, 255, 255], -1)
  cv2.putText(frame, frame_text, frame_org, font, font_scale, color, thickness)
  cv2.putText(frame, seconds_text, seconds_org, font, font_scale, color, thickness)

  # Flip image to save as its original orientation
  if flip:
     frame = cv2.flip(frame, 1)
    
  return frame

# Utility function to draw player's distance
def draw_agility(frame, flip, player_id, frame_count, fps=30):
  """
   Draws the player timer in frames and seconds on a given image frame.

   Parameters:
    frame: numpy array
        The input image frame on which the player timer is to be drawn.
    
    flip: bool
        A boolean value indicating whether the frame is to be flipped horizontally or not.
    
    player_id: int
        An integer value representing the id of the player whose timer is to be drawn.
    
    frame_count: int
        An integer value representing the frame count of the video.
    
    fps: float, optional
        A float value representing the frames per second of the video. Default value is 30 fps.

   Returns:
   numpy array
      A modified image frame with the player timer drawn on it.
  """
  
  # Get dimension with repect to right orientation
  if flip:
     frame = cv2.flip(frame, 1)

  seconds_count = frame_count/fps

  cv2.rectangle(frame, (0, 0), (550, 200), [255, 255, 255], -1)
  cv2.putText(frame, f'Player {player_id} timer (FRAMES): {frame_count}', (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  cv2.putText(frame, f'Player {player_id} timer (SECONDS): {seconds_count:.2f}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

  # Flip image to save as its original orientation
  if flip:
     frame = cv2.flip(frame, 1)
    
  return frame

def save_image(frame, flip, frame_num, root, player_id, bbox, crop_margin=20):

    xmin, ymin, xmax, ymax = bbox

    height, width = frame.shape[:2]

    # Create directory named with player_id
    os.makedirs(os.path.join(root, str(player_id)), exist_ok=True)

    # image save as name in `img_name_to_save`
    img_name_to_save = os.path.join(os.path.join(root, str(player_id)), f'{frame_num}.jpg')

    # Crop image with margin = crop_margin (20px) with respect to boundary box
    xmin_to_crop = xmin - crop_margin if xmin - crop_margin > 0 else 0
    xmax_to_crop = xmax + crop_margin if xmax + crop_margin < width else width

    ymin_to_crop = ymin - crop_margin if ymin - crop_margin > 0 else 0
    ymax_to_crop = ymax + crop_margin if ymax + crop_margin < height else height

    cropped_frame = frame[ymin_to_crop:ymax_to_crop, xmin_to_crop:xmax_to_crop]

    # Flip image to save as its original orientation
    if flip:
        cropped_frame = cv2.flip(cropped_frame, 1)

    # Write image
    cv2.imwrite(img_name_to_save, cropped_frame)

def draw_player_bbox(frame, flip, player_id, bbox, overlay=None):
    """
   Draws a bounding box and player id text on a given image frame.

   Parameters:
    frame: numpy array
        The input image frame on which the bounding box and player id text is to be drawn.
    
    flip: bool
        A boolean value indicating whether the frame is to be flipped horizontally or not.
    
    player_id: int
        An integer value representing the id of the player whose bounding box and text is to be drawn.
    
    bbox: list
        A list containing the bounding box coordinates of the player in the format [xmin, ymin, xmax, ymax].
    
    overlay: numpy array, optional
        A numpy array representing an overlay image to be blended with the input frame. Default value is None.

   Returns:
    numpy array
        A modified image frame with the bounding box and player id text drawn on it.
    """

    width = frame.shape[1]
    rectangle_color = [213, 146, 13]
    opacity = 0.4
    text_color = [247, 221, 223]
    
    # Get dimension with respect to right orientation
    if flip:
       frame = cv2.flip(frame, 1)
       bbox[0], bbox[2] = width - bbox[0], width - bbox[2]

    xmin, ymin, xmax, ymax = bbox

    w = xmax - xmin

    # Define text and font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    text = f'Player {player_id}'

    # Get text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate position of text
    text_x = int(xmin + w / 2 - text_size[0] / 2)
    text_y = int(ymin - text_size[1])

    text_psition = (text_x, text_y)

    # Calculate position and size of small rectangle
    text_rect_w = 174
    text_rect_h = 64
    text_rect_x = int(xmin + w / 2 - text_rect_w / 2)
    text_rect_y = int(ymin - text_rect_h)
    text_rect_position = ((text_rect_x, text_rect_y), (text_rect_x + text_rect_w, text_rect_y + text_rect_h))

    # Draw transparent rectangle on overlay
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), rectangle_color, thickness=3)

    cv2.rectangle(frame, *text_rect_position, rectangle_color, -1)

    # Blend overlay with original image using alpha channel
    if overlay is not None:
        frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    cv2.putText(frame, text, text_psition, font, font_scale, text_color, font_thickness)

    # Return frame to its input orientation after writing on it
    if flip:
       frame = cv2.flip(frame, 1)

    return frame


# Utility function to get player's heel position
def get_player_heel(player_mask):
  """
   Finds the heel point of a player mask.

   Parameters:
   player_mask: numpy array
      A binary numpy array representing the player mask.

   Returns:
   tuple
      A tuple containing the x and y coordinates of the heel point of the player mask.
  """

  # Get white pixels
  white_pixels_y, white_pixels_x = np.where(player_mask==1)

  white_pixels = list(zip(white_pixels_x, white_pixels_y))

  if not white_pixels:
    return

  # Get maximum point with respect to y-axis
  player_heel = max(white_pixels, key=lambda x: x[1])

  # Get maximum 700 points with respect to y-axis
  to_filter_based_on_x = sorted(white_pixels, key=lambda x: x[1], reverse=True)[:700]
  
  # Filter points that close to `maximum point with respect to y-axis` by 40px at maximum in the opposite direction of x-axis
  player_heel = list(filter(lambda x: player_heel[0] - x[0] < 40, to_filter_based_on_x))

  # Get minimum point with respect to x-axis
  player_heel = min(player_heel)

  return player_heel


# Utility function to get player's heel position
def get_player_forefoot(player_mask):
  """
   Finds the forefoot point of a player mask.

   Parameters:
   player_mask: numpy array
      A binary numpy array representing the player mask.

   Returns:
   tuple
      A tuple containing the x and y coordinates of the forefoot point of the player mask.
  """

  # Get white pixels
  white_pixels_y, white_pixels_x = np.where(player_mask==1)

  white_pixels = list(zip(white_pixels_x, white_pixels_y))

  if not white_pixels:
    return

  # Get maximum point with respect to y-axis
  player_forefoot = max(white_pixels, key=lambda x: x[1])
  
  # Get maximum 700 points with respect to y-axis
  to_filter_based_on_x = sorted(white_pixels, key=lambda x: x[1], reverse=True)[:700]

  # Filter points that close to `maximum point with respect to y-axis` by 40px at maximum in x-axis
  player_forefoot = list(filter(lambda x: x[0] - player_forefoot[0] < 40, to_filter_based_on_x))

  # Get maximum point with respect to x-axis
  player_forefoot = max(player_forefoot)

  return player_forefoot



def update_mask(frame, bbox, mask_margin=100):

    # Get frame dimensions 
    height, width = frame.shape[:2]
    
    # Extract boundary box values
    xmin, ymin, xmax, ymax = bbox

    # Create a mask with the same dimensions as the image
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Calculate padded bbox dimensions
    xmin_to_mask = xmin - mask_margin if xmin - mask_margin > 0 else 0
    xmax_to_mask = xmax + mask_margin if xmax + mask_margin < width else width

    ymin_to_mask = ymin - mask_margin if ymin - mask_margin > 0 else 0
    ymax_to_mask = ymax + mask_margin if ymax + mask_margin < height else height

    mask_start_point = (xmin_to_mask, ymin_to_mask)
    mask_end_point = (xmax_to_mask, ymax_to_mask)

    cv2.rectangle(mask, mask_start_point, mask_end_point, 255, -1)

    # Apply the mask to the original image
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return masked_frame


def default_mask(frame, start_line):
    
    """
      Updates a mask around a given bounding box in a given image frame.

      Parameters:
         frame: numpy array
            The input image frame on which the mask is to be updated.
         
         bbox: list
            A list containing the bounding box coordinates of the player in the format [xmin, ymin, xmax, ymax].
         
         mask_margin: int, optional
            An integer value representing the margin to be added to the bounding box dimensions. Default value is 100.

      Returns:
      numpy array
         A modified image frame with the updated mask around the bounding box.
    """

    height = frame.shape[0]

    # Create a mask with the same dimensions as the image
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    mask_start_point = (start_line[1][0]-150, start_line[1][1]-100)
    mask_end_point = (start_line[1][0], height)

    cv2.rectangle(mask, mask_start_point, mask_end_point, 255, -1)

    # Apply the mask to the original image
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return masked_frame

def imshow(frame, flip):
   """
      Displays a given image frame on a matplotlib window.

      Parameters:
         frame: numpy array
            The input image frame to be displayed.
         
         flip: bool
         A boolean value indicating whether the image should be flipped horizontally or not.

      Returns: None
   """
   frame_copy = frame.copy()
   frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

   if flip:
      frame_copy = cv2.flip(frame_copy, 1)

   plt.figure(figsize=(10, 7))
   plt.imshow(frame_copy)
   plt.xticks([])
   plt.yticks([])
   plt.show()
