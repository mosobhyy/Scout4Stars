import math


class Tracker:

  def __init__(self, distance_threshold=50, not_exist_threshold=10):
      self.distance_threshold = distance_threshold
      self.not_exist_threshold = not_exist_threshold
      self.track_id = 1
      self.tracked_objects = {}
      self.not_exist_timer = {}

  def reset(self):
    self.tracked_objects = {}
    self.not_exist_timer = {}

  def track(self, players_boxes):

          def assign_new_ids():
              for player in new_objects:
                  self.tracked_objects[self.track_id] = player
                  self.track_id += 1

          # Point current frame
          new_objects = []
          distances = {}

          for box in players_boxes:
              new_objects.append(box)

          # Compare between previous and current frame’s boundary boxes
          if self.tracked_objects:
              for object_id, object_pt in self.tracked_objects.copy().items():

                  object_exist = False
                  min_distance_pt = (self.distance_threshold, (0, 0))  # the closest box from previous frame
                  (xmin, ymin, xmax, ymax) = object_pt
                  center_object_pt = int((xmin+xmax)/2), int((ymin+ymax)/2)
                  # get the closest (most similar) box from previous frame
                  for new_object in new_objects:
                      (xmin, ymin, xmax, ymax) = new_object
                      center_pt = int((xmin+xmax)/2), int((ymin+ymax)/2)
                      distance = math.hypot(center_pt[0] - center_object_pt[0], center_pt[1] - center_object_pt[1])

                      if distance < min_distance_pt[0]:
                          min_distance_pt = (distance, new_object)
                          object_exist = True

                  # Update IDs position
                  if object_exist:

                      distances[object_id] = min_distance_pt[0]

                      self.tracked_objects[object_id] = min_distance_pt[1]

                      # remove point from current frame points
                      new_objects.remove(min_distance_pt[1])

                      # remove id from not_exist list
                      if object_id in self.not_exist_timer.keys():
                          self.not_exist_timer.pop(object_id)

                  # Not exist objects
                  else:
                      if object_id in self.not_exist_timer.keys():

                          # un-track objects that not exist for 10 consecutive frames
                          if self.not_exist_timer[object_id] >= self.not_exist_threshold:
                              self.tracked_objects.pop(object_id)
                              self.not_exist_timer.pop(object_id)
                          else:
                              self.not_exist_timer[object_id] += 1
                      else:
                          self.not_exist_timer[object_id] = 1

          # Assign new IDs
          assign_new_ids()

          exist_objects = {}
          for tracked_object_id, tracked_object_pt in self.tracked_objects.items():
              if tracked_object_id not in self.not_exist_timer.keys():
                  exist_objects[tracked_object_id] = tracked_object_pt

          return exist_objects, self.not_exist_timer, new_objects, distances