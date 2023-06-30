import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

class Classifier():
  def __init__(self):
    # Check if GPU is available
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    self.model = models.resnet18(pretrained=True)

    # Optimizer
    self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)

    # Loss Function
    self. loss_fn = nn.CrossEntropyLoss()

    # Change the last FC Layer to be suitable with the number of classes
    self.num_classes = None

    # Determine the transform that will be applied on the data
    self.transform =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

    

  def train(self, train_root):

    # Change the last FC Layer to be suitable with the number of classes
    self.num_classes = len(os.listdir(train_root))

    self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

    # Move model to the GPU if available
    self.model = self.model.to(self.device)

    
    self.train_ds = ImageFolder(train_root, transform=self.transform)

    # Data Loader
    self.train_loader = DataLoader(self.train_ds, batch_size = 4, shuffle=True)

    self.model.train()
    num_epochs = 100
    best_loss = 100000
    for epoch in range(num_epochs):
      running_loss = 0.0

      for images, labels in self.train_loader:
        # Move images to the GPU if possible
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(images)

        # Compute the loss
        self.loss = self.loss_fn(outputs, labels)

        # Backward pass
        self.loss.backward()  # Compute gradients

        # Update the parameters
        self.optimizer.step()

        running_loss += self.loss.item()
      print('Epoch {}/{}, Loss {}'.format(epoch + 1, num_epochs , running_loss))
      print('-' * 40)

      if self.loss < best_loss:
        best_loss = self.loss
        torch.save(self.model, 'my_model.pth')

    self.model = torch.load('my_model.pth',map_location = self.device)
    return self.model


  def get_player_class_candidates(self, test_root, train_root, model):
    # Get the number of classes
    num_players = len(os.listdir(test_root))
    num_classes = len(os.listdir(train_root))

    # Initialize the variables
    counts = np.zeros((num_players, num_classes), dtype=int)
    confidence = np.zeros((num_players, num_classes), dtype=float)

    for i, player_file_id in enumerate(sorted(os.listdir(test_root))):
      # Get full path of the player folder
      player_full_path = os.path.join(test_root, player_file_id)
      for img_name in sorted(os.listdir(player_full_path)):
        # Get path of the player's images
        img_path = os.path.join(player_full_path, img_name)

        # Read the image and apply the trtnsformation on it.
        img = Image.open(img_path)
        input_tensor = self.transform(img)

        # Add a new dimension to represent batch ->(1, 224, 224, 3)
        input_batch = input_tensor.unsqueeze(0)

        # Move the input and model to GPU if available
        input_batch = input_batch.to(self.device)
        model = model.to(self.device)

        # Use the model to predict on the images
        with torch.no_grad():
            output = model(input_batch)
        predicted_class = torch.max(output, 1).indices

        # The output has unnormalized scores. To get probabilities, we apply softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the count of the possible classes for each player and their probabilities
        counts[i][predicted_class] += 1
        confidence[i][predicted_class] += probabilities[predicted_class]


      # To normalize the sum of confidence and know the probability of each class for the player
      confidence[i]/= len(os.listdir(player_full_path))

    return counts, confidence

  def classify(self, test_root, train_root, model):
    counts, confidence = self.get_player_class_candidates(test_root, train_root, model)

    # Indicies of maximum counts
    # We gave it negative counts, because it sort ascendingly by default
    counts_indicies = np.argpartition(-counts, 1)

    # Get the two maximum class with the highest counts and confidence
    largest_elements_counts = np.zeros((counts.shape[0], 2), dtype=int)
    largest_elements_confidence = np.zeros((counts.shape[0], 2), dtype=float)
    for i in range((counts.shape[0])):
        largest_elements_counts[i][0] = counts[i][counts_indicies[i][0]]
        largest_elements_counts[i][1] = counts[i][counts_indicies[i][1]]
        largest_elements_confidence[i][0] = confidence[i][counts_indicies[i][0]]
        largest_elements_confidence[i][1] = confidence[i][counts_indicies[i][1]]

    # Check if the difference between the 2 maximum votes is less than 30, then depend on the confidence
    index_of_highest_confidence = np.zeros((counts.shape[0], 1), dtype = int)
    class_index = np.zeros((counts.shape[0], 1), dtype = int)
    for i in range(largest_elements_counts.shape[0]):
        if largest_elements_counts[i][0] - largest_elements_counts[i][1] < 30:
            index_of_highest_confidence[i] = np.argmax(largest_elements_confidence[i])
            class_index[i] = counts_indicies[i][index_of_highest_confidence[i]]
        else:
            # Since it is sorted based on their count and the difference between the top 2 class candidates is more than 30 
            # then take the class with the highest number of counts. 
            index_of_highest_confidence[i] = 0
            class_index[i] = counts_indicies[i][index_of_highest_confidence[i]]

    # To know the class that has highest counts
    return class_index
