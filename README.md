# Bangla Handwritten Character Recognition and OCR System using CNN

## 1. Project Overview

Bangla Handwritten Character Recognition and OCR System using CNN is a deep learning project that recognizes Bangla handwritten characters from isolated samples and also reconstructs handwritten words after segmenting them into individual character images. The system combines a custom convolutional neural network, a Streamlit-based OCR interface, MLflow experiment tracking, and Docker deployment for a complete end-to-end machine learning workflow.

The project is built around the BanglaLekha-Isolated dataset and targets 84 Bangla character classes, including vowels, consonants, numerals, and compound characters. It supports the following workflow:

- Draw or provide a handwritten Bangla word
- Segment the word into character-level crops
- Classify each character with a CNN
- Reconstruct the predicted character sequence
- Display Top-5 prediction probabilities for each detected character
- Track training runs and artifacts with MLflow
- Package the application with Docker for reproducible deployment

## 2. Features

- Custom CNN for 84-class Bangla character recognition
- Grayscale image preprocessing with 64x64 input normalization
- Data augmentation during training for improved generalization
- Character-level prediction with Top-5 softmax probabilities
- Word segmentation pipeline for handwritten OCR input
- Reconstructed Bangla word output from left-to-right character order
- MLflow experiment tracking for parameters, metrics, and artifacts
- Streamlit application for interactive drawing and inference
- Docker-based deployment with a lightweight Python runtime
- Training plots, confusion matrix, and model metadata export

## 3. Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Table of Contents](#3-table-of-contents)
4. [Project Structure](#4-project-structure)
5. [Dataset Description](#5-dataset-description)
6. [Character Mapping](#6-character-mapping)
7. [Data Preprocessing](#7-data-preprocessing)
8. [CNN Architecture](#8-cnn-architecture)
9. [Training Pipeline](#9-training-pipeline)
10. [MLflow Experiment Tracking](#10-mlflow-experiment-tracking)
11. [Results and Evaluation](#11-results-and-evaluation)
12. [Word Segmentation Strategy](#12-word-segmentation-strategy)
13. [Streamlit Application](#13-streamlit-application)
14. [Top-5 Probability Visualization](#14-top-5-probability-visualization)
15. [Docker Deployment](#15-docker-deployment)
16. [Installation](#16-installation)
17. [Training](#17-training)
18. [Running the Application](#18-running-the-application)
19. [Running with Docker](#19-running-with-docker)
20. [Limitations](#20-limitations)
21. [Future Improvements](#21-future-improvements)
22. [Author](#22-author)

## 4. Project Structure

```text
project/
├── app.py
├── train.py
├── requirements.txt
├── dockerfile
├── .dockerignore
├── README.md
├── labels.json
├── model_info.json
├── charechter_map.json
├── mlflow.db
├── models/
│   └── cnn_model.pth
├── plots/
│   ├── training_accuracy.png
│   ├── validation_accuracy.png
│   ├── training_loss.png
│   ├── validation_loss.png
│   ├── confusion_matrix.png
│   └── classification_report.png
```

### Key files and folders

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit OCR application with segmentation, prediction, and Top-5 output |
| `train.py` | Full training pipeline with augmentation, validation, early stopping, and MLflow logging |
| `labels.json` | Maps model output indices to dataset folder IDs |
| `charechter_map.json` | Maps dataset folder IDs to actual Bangla Unicode characters |
| `model_info.json` | Stores model metadata such as architecture and accuracy summary |
| `models/cnn_model.pth` | Best saved model checkpoint |
| `plots/` | Training and evaluation figures |
| `artifacts/` | MLflow-related outputs and saved experiment artifacts |

## 5. Dataset Description

### BanglaLekha-Isolated

The project uses the BanglaLekha-Isolated dataset, a folder-organized handwritten character dataset designed for isolated Bangla character recognition. It contains 84 classes covering:

- Bangla vowels
- Bangla consonants
- Bangla numerals
- Compound characters and ligatures

### Why this dataset was chosen

- It is directly aligned with the task of isolated character recognition
- The folder-based class structure is convenient for supervised learning
- It provides a broad 84-class label space suitable for Bangla OCR research
- The dataset contains handwritten samples, which matches the real-world OCR use case of this project

### Folder organization concept

The dataset is arranged as one folder per class. During training, each folder name is treated as a label, and each image inside that folder is a sample for that class.

| Dataset aspect | Description |
| --- | --- |
| Organization | Folder-based |
| Number of classes | 84 |
| Sample type | Handwritten isolated characters |
| Input format | Image files under class-specific directories |

## 6. Character Mapping

This project uses a two-step mapping strategy to convert model outputs into readable Bangla characters:

1. The CNN predicts a class index.
2. `labels.json` converts that index into a dataset folder ID.
3. `charechter_map.json` converts the folder ID into the final Bangla Unicode character.

This design keeps the model training labels separate from the human-readable script representation.

| Artifact | Role | Example |
| --- | --- | --- |
| `labels.json` | Model index to class ID mapping | `0 -> 1`, `1 -> 10` |
| `charechter_map.json` | Class ID to Bangla character mapping | `1 -> অ`, `12 -> ক` |

This mapping approach is especially useful when the dataset folder IDs are numeric but the final application must show Bangla characters to the user.

## 7. Data Preprocessing

The preprocessing pipeline standardizes handwritten images before they are passed to the CNN. It ensures that training and inference use a consistent representation.

### Preprocessing steps

1. Image loading
2. Grayscale processing
3. Resizing to 64x64
4. Tensor conversion
5. Normalization
6. Data augmentation
7. Train-validation split

### Preprocessing flow diagram

```text
Input Image
	|
	v
Image Loading
	|
	v
Grayscale Conversion
	|
	v
Resize to 64x64
	|
	v
Tensor Conversion
	|
	v
Normalization to [-1, 1]
	|
	v
Train Augmentation / Validation Standardization
	|
	v
80/20 Train-Validation Split
```

### Detailed explanation

- **Image loading**: Images are read from the dataset folders and opened with Pillow.
- **Grayscale processing**: Each sample is converted to a single channel image to match the CNN input.
- **Resizing to 64x64**: Every image is resized to a fixed spatial resolution so the model sees a uniform input size.
- **Tensor conversion**: The image is converted to a PyTorch tensor.
- **Normalization**: Pixel values are normalized using mean `0.5` and standard deviation `0.5`, which maps inputs to approximately `[-1, 1]`.
- **Data augmentation**: The training pipeline applies random rotation, affine transforms, and perspective distortion to improve robustness.
- **Train-validation split**: The dataset is split into 80% training and 20% validation using a fixed random seed for reproducibility.

### Training-time augmentation used in the codebase

| Augmentation | Purpose |
| --- | --- |
| Random rotation | Improves robustness to handwriting orientation |
| Random affine transform | Simulates translation and scale variation |
| Random perspective transform | Helps the model handle slight slant and perspective effects |

## 8. CNN Architecture

The project uses a custom convolutional neural network designed for grayscale 64x64 input images.

### Architecture summary

- Input: `1 x 64 x 64` grayscale image
- Four convolutional feature extraction blocks
- Batch normalization after each convolution
- ReLU activations for non-linearity
- Max pooling after each block
- Fully connected classifier with dropout
- Output layer with 84 classes

### ASCII architecture diagram

```text
Input: 1 x 64 x 64
	|
	v
Conv2D(1 -> 32) -> BatchNorm -> ReLU -> MaxPool
	|
	v
Conv2D(32 -> 64) -> BatchNorm -> ReLU -> MaxPool
	|
	v
Conv2D(64 -> 128) -> BatchNorm -> ReLU -> MaxPool
	|
	v
Conv2D(128 -> 256) -> BatchNorm -> ReLU -> MaxPool
	|
	v
Flatten
	|
	v
Linear(4096 -> 512) -> ReLU -> Dropout(0.5)
	|
	v
Linear(512 -> 84)
	|
	v
Softmax probabilities
```

### Layer-by-layer specification

| Stage | Layers | Output intent |
| --- | --- | --- |
| Block 1 | Conv2D(1, 32), BatchNorm, ReLU, MaxPool | Extract low-level stroke features |
| Block 2 | Conv2D(32, 64), BatchNorm, ReLU, MaxPool | Capture edges and character parts |
| Block 3 | Conv2D(64, 128), BatchNorm, ReLU, MaxPool | Learn richer shape representations |
| Block 4 | Conv2D(128, 256), BatchNorm, ReLU, MaxPool | Build high-level discriminative features |
| Classifier | Flatten, Linear(4096, 512), ReLU, Dropout(0.5), Linear(512, 84) | Produce class logits |

### Model size

| Metric | Value |
| --- | --- |
| Model class | `BanglaCNN` |
| Number of classes | 84 |
| Total parameters | 2,529,556 |
| Trainable parameters | 2,529,556 |

## 9. Training Pipeline

The training script implements a full supervised learning pipeline for Bangla handwritten character classification.

### Training workflow diagram

```text
Dataset Loading
	|
	v
Label Mapping
	|
	v
Data Augmentation and Batch Generation
	|
	v
Forward Propagation
	|
	v
Loss Computation
	|
	v
Backpropagation
	|
	v
Optimization with Adam
	|
	v
Validation Loop
	|
	v
Scheduler and Early Stopping
	|
	v
Best Checkpoint Saving
	|
	v
Model and Artifact Export
```

### Training stages

1. **Dataset loading**: The dataset is loaded from `BanglaLekha-Isolated/Images` using a custom PyTorch `Dataset`.
2. **Label mapping**: Folder names are mapped to class indices, and those indices are later converted back to readable Bangla characters.
3. **Batch generation**: `DataLoader` instances create shuffled training batches and deterministic validation batches.
4. **Forward propagation**: Images pass through the CNN to produce class logits.
5. **Loss computation**: `CrossEntropyLoss` is used with class weights to reduce the impact of class imbalance.
6. **Backpropagation**: Gradients are computed and propagated through the network.
7. **Optimization**: Adam updates model parameters.
8. **Validation**: After every epoch, the validation loader is used to measure generalization performance.
9. **Checkpoint saving**: The best validation model is saved to `models/cnn_model.pth`.
10. **Model export**: The script saves `labels.json`, `model_info.json`, and evaluation plots for downstream inference and reporting.

### Training configuration used in the codebase

| Setting | Value |
| --- | --- |
| Default epochs | 50 |
| Default batch size | 64 |
| Default learning rate | 0.001 |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau |
| Early stopping | Enabled |
| Loss function | Weighted CrossEntropyLoss |
| Precision | Mixed precision on CUDA |

### Reproducibility details

- The train-validation split uses a fixed seed of `42`.
- The model checkpoint is saved only when validation accuracy improves.
- The training script logs the active device and hyperparameters to MLflow.

## 10. MLflow Experiment Tracking

This project uses MLflow to make experiments reproducible and auditable.

### What is tracked

- Experiment creation
- Parameter logging
- Metric logging
- Artifact logging
- Model checkpoint tracking

### MLflow workflow

1. The training script sets the tracking URI through `MLFLOW_TRACKING_URI`.
2. A new experiment named `Bangla OCR Training` is created or reused.
3. Hyperparameters such as epochs, batch size, learning rate, optimizer, scheduler, and early stopping patience are logged.
4. Training and validation metrics are logged at every epoch.
5. Model checkpoint, label map, metadata, and plots are logged as artifacts.

### Logged parameters

| Parameter | Description |
| --- | --- |
| `epochs` | Number of training epochs |
| `batch_size` | Mini-batch size used by the DataLoader |
| `learning_rate` | Initial Adam learning rate |
| `optimizer` | Optimization algorithm |
| `lr_scheduler` | Learning-rate scheduling strategy |
| `early_stopping_patience` | Early stopping patience |
| `device` | CPU or CUDA execution target |

### Logged metrics

| Metric | Description |
| --- | --- |
| `train_loss` | Average training loss per epoch |
| `train_accuracy` | Training accuracy per epoch |
| `val_loss` | Validation loss per epoch |
| `val_accuracy` | Validation accuracy per epoch |
| `learning_rate` | Current scheduler-adjusted learning rate |

### Logged artifacts

| Artifact | Purpose |
| --- | --- |
| `models/cnn_model.pth` | Best model checkpoint |
| `labels.json` | Output index-to-class mapping |
| `model_info.json` | Model metadata and summary statistics |
| `plots/training_loss.png` | Loss curve |
| `plots/training_accuracy.png` | Accuracy curve |
| `plots/confusion_matrix.png` | Class-level evaluation matrix |

## 11. Results and Evaluation

The evaluation package is designed to show both training dynamics and class-level behavior. The saved model metadata indicates a final training accuracy of `0.9132` and a best validation accuracy of `0.9335`.

### Training Accuracy

![Training Accuracy](https://chatgpt.com/c/plots/training_accuracy.png)

This curve shows how quickly the model learns discriminative character features during training. A steadily rising trend indicates that the CNN is fitting the handwriting patterns effectively and converging toward a stable solution.

### Validation Accuracy

![Validation Accuracy](https://chatgpt.com/c/plots/validation_accuracy.png)

This curve reflects the model's generalization capability on unseen data. It is especially useful for detecting overfitting when compared against the training accuracy curve.

### Training Loss

![Training Loss](https://chatgpt.com/c/plots/training_loss.png)

Training loss measures optimization progress. A decreasing loss curve indicates that the classifier is improving its predictions and minimizing the cross-entropy objective over time.

### Validation Loss

![Validation Loss](https://chatgpt.com/c/plots/validation_loss.png)

Validation loss is used to judge whether the model remains stable on unseen samples. It is also a useful signal for learning-rate scheduling and checkpoint selection.

### Confusion Matrix

![Confusion Matrix](https://chatgpt.com/c/plots/confusion_matrix.png)

The confusion matrix provides class-level performance insight. It helps identify which Bangla characters are commonly confused and where the model may need more data, better augmentation, or architectural changes.

### Classification Report

![Classification Report](https://chatgpt.com/c/plots/classification_report.png)

The classification report summarizes precision, recall, F1 score, and support for each class. It is useful for identifying whether the model performs consistently across common and rare characters.

### Evaluation Summary

- The model learns meaningful character-level features with a compact custom CNN.
- Validation tracking helps verify that generalization remains strong during training.
- The confusion matrix and classification report are essential for class-by-class error analysis.
- The best checkpoint is selected using validation accuracy rather than training accuracy.

## 12. Word Segmentation Strategy

This section describes the OCR segmentation pipeline used before classification. It is one of the most important parts of the system because handwritten Bangla words often contain connected strokes, headline structures, and visually complex character groups.

### Segmentation pipeline

1. User draws a word
2. Canvas image acquisition
3. RGB to grayscale conversion
4. Binary thresholding
5. Morphological cleaning
6. Contour detection
7. Bounding box extraction
8. Noise filtering
9. Character cropping
10. Left-to-right sorting
11. Individual character prediction
12. Word reconstruction

### ASCII segmentation diagram

```text
Input Word
	|
	v
Thresholding
	|
	v
Morphological Operations
	|
	v
Contour Detection
	|
	v
Bounding Box Extraction
	|
	v
Character Cropping
	|
	v
CNN Classification
	|
	v
Word Reconstruction
```

### How the current implementation works

The Streamlit canvas returns an RGBA image. The OCR pipeline then converts it into RGB, converts the result to grayscale, and applies binary inversion so that the handwritten strokes become the foreground. From there, the code uses a projection-based strategy to estimate the Bangla headline region and suppress it temporarily, which helps separate connected characters.

After the initial cleanup, the system performs vertical projection analysis to identify candidate character boundaries. Each candidate region is refined using contour detection and a bounding-box filter. Small noisy regions are discarded, and the remaining character crops are sorted from left to right before classification.

### Detailed step-by-step explanation

#### 1. User draws a word
The user writes a Bangla word directly on the Streamlit canvas.

#### 2. Canvas image acquisition
The canvas output is captured as a NumPy array and passed to the segmentation pipeline.

#### 3. RGB to grayscale conversion
The canvas image is converted from RGBA to RGB, then to grayscale so that intensity-based segmentation is easier.

#### 4. Binary thresholding
Thresholding converts the grayscale image into a binary representation where the handwriting becomes easier to isolate from the background.

#### 5. Morphological cleaning
In a generic OCR pipeline, this step removes isolated noise and closes small gaps. In the current codebase, this role is approximated through binary inversion, headline suppression, contour refinement, and size filtering rather than a separate morphological kernel call.

#### 6. Contour detection
Contours are detected around foreground regions to identify character-like blobs.

#### 7. Bounding box extraction
Each contour is wrapped in a rectangular bounding box, which provides a clean crop for classification.

#### 8. Noise filtering
Very small bounding boxes are discarded so that ink specks and scan artifacts do not become false character predictions.

#### 9. Character cropping
Each bounding box is padded and cropped from the source image to preserve character context.

#### 10. Left-to-right sorting
All detected crops are sorted by their x-coordinate so the final output preserves the natural reading order.

#### 11. Individual character prediction
Each crop is resized to 64x64, normalized, and fed into the CNN for classification.

#### 12. Word reconstruction
The predicted characters are concatenated in order to reconstruct the final Bangla word.

### Practical OCR notes

- The segmentation strategy is designed for handwritten input and may be sensitive to how the user spaces letters.
- The pipeline is strongest when the user writes clearly and keeps the word inside the canvas.
- The left-to-right sort is essential for reconstructing a readable character sequence.

## 13. Streamlit Application

The Streamlit application provides the interactive OCR front end. It lets the user draw a Bangla word, segment it, classify each character, and inspect the confidence scores for the detected characters.

### What the UI includes

- Drawing canvas for handwritten input
- Character segmentation preview
- Per-character prediction cards
- Word reconstruction output
- Confidence scores for each prediction
- Top-5 probability visualization for each character

### Application workflow

1. The user draws a word on the canvas.
2. The application captures the drawn image.
3. The segmentation module extracts individual character crops.
4. The CNN predicts each character.
5. The application reconstructs the final word.
6. The UI displays the Top-5 probabilities for interpretation.

## 14. Top-5 Probability Visualization

The application does not stop at the most likely class. Instead, it shows the five most probable predictions for each segmented character.

### Why Top-5 probabilities matter

- They provide a better sense of model uncertainty
- They help interpret confusing handwriting samples
- They show whether the top prediction is clearly dominant or only slightly ahead of alternatives

### Visualization method

- The model outputs logits for all 84 classes
- Softmax converts logits into normalized probabilities
- The top five probabilities are selected with `topk`
- Plotly renders the results as an interactive bar chart

### Confidence interpretation

| Confidence pattern | Interpretation |
| --- | --- |
| One class dominates the Top-5 list | The model is confident |
| Several classes are close together | The handwriting is ambiguous or visually similar |
| All probabilities are low | The sample may be noisy, poorly segmented, or out of distribution |

### Plotly chart output

The bar chart displays the predicted Bangla character on the x-axis and the probability on the y-axis. This is useful for both debugging and academic demonstration because it explains not just what the model predicted, but how certain it was.

## 15. Docker Deployment

Docker is used to package the OCR application into a reproducible container with all dependencies preinstalled.

### Purpose of the Dockerfile

The Docker build recipe installs the Python dependencies, system packages required by OpenCV and Streamlit, copies the application code into the container, and starts the Streamlit server on port 8501.

### Containerization workflow

1. Build a Docker image from the project files.
2. Install runtime dependencies inside the container.
3. Copy the trained model and supporting JSON artifacts.
4. Start the Streamlit app inside the container.
5. Map container port 8501 to the host so the UI is accessible in the browser.

### Build and run commands

```bash
docker build -t bangla-ocr .
docker run -p 8501:8501 bangla-ocr
```

### Deployment details from the repository

| Item | Details |
| --- | --- |
| Base image | `python:3.11-slim` |
| Exposed port | `8501` |
| Main entrypoint | `streamlit run app.py` |
| Runtime dependencies | Python packages from `requirements.txt` |
| System dependencies | `gcc`, `libgl1`, `libglib2.0-0` |

### Important note about file naming

The repository currently stores the Docker recipe as `dockerfile` in lowercase. If you want to use the exact build command above without specifying a custom file name, rename it to `Dockerfile`. Otherwise, build with `docker build -f dockerfile -t bangla-ocr .`.


## 16. Installation

### Prerequisites

- Python environment capable of running PyTorch and Streamlit
- Dataset extracted into `BanglaLekha-Isolated/Images`
- Optional: Docker for containerized deployment

### Install dependencies

```bash
git clone <your-repository-url>
cd project-assignment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify required artifacts

Make sure the following files exist before running the application:

- `models/cnn_model.pth`
- `labels.json`
- `model_info.json`
- `charechter_map.json`

## 17. Training

Use the training script to train the CNN from scratch or to reproduce the reported experiment.

### Default training command

```bash
python train.py
```

### Custom training command

```bash
python train.py --epochs 50 --batch-size 64 --lr 0.001 --patience 7
```

### What training does

- Loads the BanglaLekha-Isolated dataset
- Applies augmentation to the training split
- Computes class weights for imbalance handling
- Trains the CNN with Adam and mixed precision on CUDA when available
- Uses validation accuracy for early stopping and checkpoint selection
- Saves the best checkpoint to `models/cnn_model.pth`
- Exports `labels.json`, `model_info.json`, and evaluation plots
- Logs everything to MLflow

### MLflow tracking URI

You can override the default MLflow backend by setting:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

## 18. Running the Application

Start the Streamlit app locally after the model artifacts have been prepared.

```bash
streamlit run app.py
```

### What happens in the app

- A drawing canvas appears in the browser
- The user writes a Bangla word
- The app segments the drawing into characters
- Each character is classified by the CNN
- The predicted word is reconstructed and displayed
- Top-5 predictions are shown for each detected character

## 19. Running with Docker

Run the application inside a container for a reproducible environment.

### Build the image

```bash
docker build -t bangla-ocr .
```

### Start the container

```bash
docker run -p 8501:8501 bangla-ocr
```

### Deployment validation checklist

- The image is created successfully
- The container starts without dependency errors
- Port 8501 is available in the browser
- The OCR canvas loads and responds to predictions

## 20. Limitations

- Handwriting variability can reduce recognition accuracy when characters are written in unusual styles.
- Bangla character ambiguity may cause confusion between visually similar classes.
- Compound character complexity makes segmentation harder than isolated classification.
- The current segmentation pipeline can struggle when characters overlap heavily or when the word contains irregular spacing.
- Characters with weak strokes or excessive noise may be filtered out during bounding-box cleanup.

## 21. Future Improvements

- Replace the custom CNN with stronger backbones such as ResNet architectures
- Evaluate Vision Transformer based image encoders
- Explore CRNN-based OCR for sequence-aware recognition
- Improve segmentation with more robust preprocessing and layout analysis
- Train on a larger and more diverse Bangla handwritten dataset
- Move toward an end-to-end OCR pipeline that reduces dependence on manual segmentation

## 22. Author

| Field | Details |
| --- | --- |
| Name | Md. Reshad Romim Khan |
| Course | Certificate on Machine Learning / Deep Learning |
| Email | reshadromim013@gmail.com |

