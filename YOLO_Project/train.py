"""
Main file for training Yolo model on Pascal VOC dataset
"""

# Import necessary libraries
import torch  # PyTorch for deep learning operations
import torchvision.transforms as transforms  # For applying image transformations
import torch.optim as optim  # PyTorch optimizers for training models
import torchvision.transforms.functional as FT  # For applying functional image transformations
from tqdm import tqdm  # Progress bar for visualizing training progress
from torch.utils.data import DataLoader  # For creating batches of data during training
from model import Yolov1  # YOLOv1 model definition from external file (model.py)
from dataset import VOCDataset  # Custom dataset for Pascal VOC dataset (dataset.py)
from utils import (
    non_max_suppression,  # Function for NMS to filter out redundant bounding boxes
    mean_average_precision,  # Metric to evaluate detection performance
    intersection_over_union,  # Metric for evaluating bounding box overlap
    cellboxes_to_boxes,  # Converts YOLO model's grid output to bounding boxes
    get_bboxes,  # Function to extract bounding boxes from model output
    plot_image,  # Function to plot images and visualize results
    save_checkpoint,  # Function to save model checkpoint
    load_checkpoint,  # Function to load a saved model checkpoint
)
from loss import YoloLoss  # Custom YOLO loss function (loss.py)

# Setting random seed for reproducibility of experiments
seed = 123
torch.manual_seed(seed)  # Set the random seed for PyTorch operations

# Hyperparameters and training configurations
LEARNING_RATE = 2e-5  # Learning rate for the optimizer (small value for fine-tuning)
DEVICE = "cpu"  # Set device to CPU by default, change to "cuda" for GPU if available
BATCH_SIZE = 16  # Batch size for training, 16 is chosen for memory efficiency
WEIGHT_DECAY = 0  # Weight decay for regularization (no regularization here)
EPOCHS = 1000  # Number of training epochs
NUM_WORKERS = 2  # Number of worker threads to load data in parallel
PIN_MEMORY = True  # Pin memory for faster data transfer to GPU (if used)
LOAD_MODEL = False  # Flag to indicate whether to load an existing model checkpoint
LOAD_MODEL_FILE = "overfit.pth.tar"  # File to load the saved model checkpoint (if LOAD_MODEL=True)
IMG_DIR = "data/images"  # Directory where the input images are stored
LABEL_DIR = "data/labels"  # Directory where the label files (annotations) are stored


# Compose class to chain multiple image transformations
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms  # Store the list of transformations

    def __call__(self, img, bboxes):
        # Apply each transformation in the list to the image and its bounding boxes
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes  # Return transformed image and bounding boxes


# Define the image transformations to be applied to each training sample
transform = Compose([
    transforms.Resize((448, 448)),  # Resize image to 448x448 pixels (standard input size for YOLO)
    transforms.ToTensor(),  # Convert the image to a tensor for processing by the model
])

# Function to train the model for one epoch
def train_fn(train_loader, model, optimizer, loss_fn):
    # Create a progress bar for the training loop using tqdm
    loop = tqdm(train_loader, leave=True)
    mean_loss = []  # List to keep track of the loss for each batch

    # Iterate over each batch in the DataLoader
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)  # Move the input and target to the device (CPU or GPU)

        # Forward pass: pass the input through the model
        out = model(x)

        # Calculate the loss by comparing the model output with the ground truth labels
        loss = loss_fn(out, y)

        # Append the loss for this batch to the mean_loss list
        mean_loss.append(loss.item())

        # Zero out the gradients before the backward pass
        optimizer.zero_grad()

        # Backward pass: compute the gradients
        loss.backward()

        # Update the model weights using the optimizer
        optimizer.step()

        # Update the progress bar with the current batch loss
        loop.set_postfix(loss=loss.item())  # Display the loss for the current batch

    # Return the average loss for the entire epoch
    return mean_loss


# Function to validate the model (optional but useful for monitoring overfitting)
def eval_fn(val_loader, model, loss_fn):
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm, etc.)
    mean_loss = []

    # Disable gradient computation for validation to save memory and computation
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)  # Get the model predictions

            # Compute the loss
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())  # Append the loss to the list

    # Return the average validation loss
    return mean_loss


# Function to load the dataset using the custom VOCDataset class
def load_data():
    # Define the dataset and apply the transformations
    train_dataset = VOCDataset(IMG_DIR, LABEL_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)

    return train_loader


# Main training loop (called when script is executed)
def main():
    # Initialize the YOLOv1 model and move it to the appropriate device
    model = Yolov1().to(DEVICE)

    # Define the optimizer (using Adam here, but other optimizers like SGD can also be used)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Define the loss function for YOLOv1 (custom loss function for object detection)
    loss_fn = YoloLoss()

    # Optionally, load an existing model checkpoint to continue training
    if LOAD_MODEL:
        load_checkpoint(LOAD_MODEL_FILE, model, optimizer)

    # Load training data
    train_loader = load_data()

    # Loop through the epochs
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        # Train for one epoch
        mean_train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        avg_train_loss = sum(mean_train_loss) / len(mean_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Save a model checkpoint after every epoch
        save_checkpoint(model, optimizer, f"checkpoint_epoch_{epoch+1}.pth")

# Start training if this script is executed
if __name__ == "__main__":
    main()
