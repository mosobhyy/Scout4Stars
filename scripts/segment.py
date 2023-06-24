import os

# Install iglovikov_helper_functions package
os.system("pip install iglovikov_helper_functions")

# Install people_segmentation package
os.system("pip install people_segmentation > /dev/null")

import cv2
import albumentations as albu
import numpy as np
import torch
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from people_segmentation.pre_trained_models import create_model

class Segmenter:
    def __init__(self, model=None, transform=None):

        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = create_model("Unet_2020-07-20") if not model else create_model(model)
        self.model.to(self.device)
        self.model.eval()
        # Define the image transformation pipeline
        self.transform = albu.Compose([albu.Normalize(p=1)], p=1) if not transform else transform

    def segment(self, img):
      # Apply padding to the img and transform it
      padded_img, pads = pad(img, factor=32, border=cv2.BORDER_CONSTANT)
      x = self.transform(image=padded_img)["image"]
      x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

      # Move the tensors to the GPU if available
      x = x.to(self.device)

      # Make predictions using the segmentation model
      with torch.no_grad():
          # Ensure that the block is executed on the GPU
          with torch.cuda.amp.autocast():
              prediction = self.model(x)[0][0]

      # Move the predicted mask to the CPU
      prediction = prediction.cpu()

      # Post-process the predicted mask and remove padding
      mask = (prediction > 0).numpy().astype(np.uint8)
      mask = unpad(mask, pads)

      # Apply the mask to the original img
      masked_img = cv2.bitwise_and(img, img, mask=mask)

      # Apply the mask to the original img
      dst = cv2.addWeighted(img, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)

      return mask, masked_img, dst