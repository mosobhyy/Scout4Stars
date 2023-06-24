import os

# Clone the repository
os.system("git clone https://github.com/ultralytics/yolov5")

# Move into the cloned directory
os.chdir("yolov5")

# Install the requirements
os.system("pip install -qr requirements.txt")

# Move back to root directory
os.chdir("..")

import torch

class Detector:
    def __init__(self,
                 weights=('player_weights.pt', 'ball_weights.pt', 'cone_weights.pt', 'flag_weights.pt'),
                 classes=('player', 'ball', 'cone', 'flag')):
        if len(weights) != len(classes):
            raise ValueError('The number of weights must be equal to the number of classes.')
        
        self.weights = weights
        self.classes = classes
        
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.detectors_dict = {}
        for weight_path, class_name in zip(weights, classes):
            self.detectors_dict[class_name] = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path).to(self.device)
    
    def detect(self, img, target_class=None):
        # If no target class, apply detection for all classes
        if not target_class:
            total_result = torch.tensor([]).to(self.device)
            for class_name, detector in self.detectors_dict.items():
              result = detector(img).xyxy[0]
              # Concatenate the tensors along the first dimension
              total_result = torch.cat((total_result, result), dim=0)
                    
        # Check if target class does not exist
        elif target_class.lower() not in self.classes:
            raise ValueError('Target class does not exist!')

        # If target claass
        else:
          total_result = self.detectors_dict[target_class](img).xyxy[0]
        
        return total_result

















