"""
train.py

This script handles the complete training pipeline for the Bangla Handwritten
Character Recognition model. It performs the following steps:
1.  Loads the BanglaLekha-Isolated dataset.
2.  Applies data augmentation to the training set.
3.  Calculates class weights to handle data imbalance.
4.  Defines and initializes a deep CNN model.
5.  Implements a full training loop with:
    - Automatic CUDA device selection.
    - Mixed-precision training (torch.amp) on CUDA.
    - Adam optimizer and ReduceLROnPlateau learning rate scheduler.
    - Early stopping to prevent overfitting.
6.  Saves only the best model based on validation accuracy.
7.  Tracks experiments using MLflow, logging parameters, metrics, and artifacts.
8.  Generates and saves training plots (loss, accuracy) and a confusion matrix.
9.  Saves model artifacts (`model.pth`, `labels.json`, `model_info.json`).

The script is configurable via command-line arguments and environment variables.
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import mlflow

# --- Configuration ---
# Use environment variable for MLflow tracking URI, default to local file storage
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# --- Pathlib-based Path Definitions ---
ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "BanglaLekha-Isolated" / "Images" # Assumed dataset location
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Create necessary directories
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)


# --- Custom Dataset Class ---
class BanglaLekhaDataset(Dataset):
    """Custom PyTorch Dataset for the BanglaLekha-Isolated dataset."""
    def __init__(self, root_dir: Path):
        """
        Args:
            root_dir (Path): Path to the dataset directory.
        """
        self.root_dir = root_dir
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]

        if not self.classes:
            raise FileNotFoundError(
                f"No character subdirectories found in {self.root_dir}. "
                "Please ensure the dataset is downloaded and extracted correctly."
            )

        for cls_name in self.classes:
            class_idx = self.class_to_idx[cls_name]
            class_dir = self.root_dir / cls_name

            for ext in extensions:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img_path, class_idx = self.samples[idx]
        try:
            # Open image, convert to grayscale to ensure 1 channel
            image = Image.open(img_path).convert("L")
            return image, class_idx
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a dummy image and a valid index to avoid crashing the loader
            return Image.new("L", (64, 64)), class_idx


# --- CNN Model Definition ---
class BanglaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Early Stopping Class ---
class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, verbose: bool = False, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.inf

    def __call__(self, val_acc: float, model: nn.Module, model_path: Path):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_acc: float, model: nn.Module, model_path: Path):
        """Saves model when validation accuracy increases."""
        if self.verbose:
            logging.info(f'Validation accuracy increased ({self.val_acc_max:.4f} --> {val_acc:.4f}). Saving model...')
        torch.save(model.state_dict(), model_path)
        self.val_acc_max = val_acc


# --- Utility Functions ---
class SubsetWrapper(Dataset):
    """
    Wrapper for a PyTorch Subset to apply a transform.
    """
    def __init__(self, subset: Subset, transform: Any = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.subset)


def get_data_loaders(
    dataset_path: Path, batch_size: int
) -> Tuple[DataLoader, DataLoader, Dict[int, str], int, List[int]]:
    """Creates and returns data loaders for train and validation sets."""

    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1]
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create dataset
    full_dataset = BanglaLekhaDataset(root_dir=dataset_path)
    num_classes = len(full_dataset.classes)
    labels_map = full_dataset.idx_to_class

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Use a fixed generator for reproducible splits
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms via wrapper
    train_dataset = SubsetWrapper(train_subset, transform=train_transform)
    val_dataset = SubsetWrapper(val_subset, transform=val_transform)

    # Dynamic num_workers
    num_workers = min(8, os.cpu_count() or 1)
    logging.info(f"Using {num_workers} workers for data loading.")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Get training labels for weight calculation
    train_labels = [full_dataset.samples[i][1] for i in train_subset.indices]

    return train_loader, val_loader, labels_map, num_classes, train_labels


def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Calculates class weights based on their frequency from a list of labels."""
    class_counts = np.zeros(num_classes)
    for label in labels:
        class_counts[label] += 1

    # Handle zero counts to avoid division by zero
    class_counts[class_counts == 0] = 1

    weights = 1.0 / class_counts
    weights = weights / np.sum(weights) # Normalize
    return torch.from_numpy(weights).float()


def plot_and_save(
    data: List[float],
    val_data: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path
):
    """Generates and saves a plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(data, label=f'Training {ylabel}')
    plt.plot(val_data, label=f'Validation {ylabel}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Path
):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --- Main Training Function ---
def train(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int
):
    """Main function to run the training pipeline."""
    
    # --- MLflow Setup ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Bangla OCR Training")

    with mlflow.start_run() as run:
        logging.info(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "lr_scheduler": "ReduceLROnPlateau",
            "early_stopping_patience": patience
        })

        # --- Device Setup ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        mlflow.log_param("device", device)

        # --- Data Loading ---
        try:
            train_loader, val_loader, labels_map, num_classes, train_labels = get_data_loaders(DATASET_PATH, batch_size)
        except FileNotFoundError as e:
            logging.error(e)
            logging.error("Please download the 'BanglaLekha-Isolated' dataset and place it in the project root.")
            return

        # --- Model, Loss, Optimizer, Scheduler ---
        model = BanglaCNN(
            num_classes=num_classes
        ).to(device)
        
        # Calculate class weights for handling imbalance
        class_weights = calculate_class_weights(train_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=3,
            factor=0.1
)
        
        # --- Mixed Precision Setup ---
        if device == "cuda":
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.amp.GradScaler(enabled=False)

        # --- Early Stopping ---
        model_path = MODELS_DIR / "cnn_model.pth"
        early_stopper = EarlyStopping(patience=patience, verbose=True)

        # --- Training History ---
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

        # --- Training Loop ---
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(predicted == labels).sum().item() / labels.size(0):.4f}'
                })

            train_loss = running_loss / len(train_loader)
            train_acc = correct_train / total_train
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            all_preds = []
            all_labels = []

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{(predicted == labels).sum().item() / labels.size(0):.4f}'
                    })

            val_loss_avg = val_loss / len(val_loader)
            val_acc = correct_val / total_val
            history['val_loss'].append(val_loss_avg)
            history['val_acc'].append(val_acc)

            logging.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss_avg, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

            # Scheduler and Early Stopping
            scheduler.step(val_loss_avg)
            early_stopper(val_acc, model, model_path)
            if early_stopper.early_stop:
                logging.info("Early stopping triggered.")
                break
        
        logging.info("Finished Training.")

        # --- Save Artifacts ---
        # Save labels map
        labels_path = ROOT_DIR / "labels.json"
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(labels_map, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved labels map to {labels_path}")

        # Save model info
        model_info_path = ROOT_DIR / "model_info.json"
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_info = {
            "model_class": model.__class__.__name__,
            "num_classes": num_classes,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "final_train_accuracy": history['train_acc'][-1],
            "best_val_accuracy": early_stopper.val_acc_max,
        }
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        logging.info(f"Saved model info to {model_info_path}")

        # --- Generate and Save Plots ---
        loss_plot_path = PLOTS_DIR / "training_loss.png"
        plot_and_save(history['train_loss'], history['val_loss'], 'Training and Validation Loss', 'Epochs', 'Loss', loss_plot_path)
        logging.info(f"Saved loss plot to {loss_plot_path}")

        acc_plot_path = PLOTS_DIR / "training_accuracy.png"
        plot_and_save(history['train_acc'], history['val_acc'], 'Training and Validation Accuracy', 'Epochs', 'Accuracy', acc_plot_path)
        logging.info(f"Saved accuracy plot to {acc_plot_path}")

        cm_plot_path = PLOTS_DIR / "confusion_matrix.png"
        plot_confusion_matrix(all_labels, all_preds, list(labels_map.values()), cm_plot_path)
        logging.info(f"Saved confusion matrix to {cm_plot_path}")

        # --- Log Artifacts to MLflow ---
        logging.info("Logging artifacts to MLflow...")
        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(labels_path))
        mlflow.log_artifact(str(model_info_path))
        mlflow.log_artifact(str(loss_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(acc_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(cm_plot_path), artifact_path="plots")
        
        logging.info("Training pipeline complete.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train a CNN for Bangla Character Recognition.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--patience", type=int, default=7, help="Patience for early stopping.")
    
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )
