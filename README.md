# PyTorch Neural Network Tutorial

This repository contains a Jupyter notebook demonstrating how to train a simple neural network using PyTorch. The example uses the two-moons dataset for binary classification.

## Setup Instructions

### 1. Create a Conda Environment

```bash
# Create a new conda environment named 'torchenv'
conda create -n torchenv python=3.13

# Activate the environment
conda activate torchenv
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook
```

Then open `notebook.ipynb` in your browser.

## Project Structure

- `notebook.ipynb`: Main Jupyter notebook containing the PyTorch tutorial
- `requirements.txt`: List of required Python packages

## Features

- Data preprocessing with scikit-learn
- PyTorch neural network implementation
- Training loop with loss and accuracy tracking
- Model evaluation on test set

## Requirements

- Python 3.13
- PyTorch
- scikit-learn
- Jupyter Notebook

## Note

Make sure you have CUDA installed on your system if you want to use GPU acceleration with PyTorch. The requirements.txt file will automatically install the appropriate PyTorch version for your system.

---

# Assignment: MNIST Classification

## Assignment Overview

The assignment consists of two main parts:

1. **Model Implementation and Training** (`run_experiment_mnist.py`):
   - Use `run_experiment_moons.py` as a reference to implement the MNIST version
   - Implement the training and evaluation process for MNIST digit classification
   - Track and report training/validation metrics
   - Save the trained model

2. **Model Visualization and Analysis** (`visualization_mnist.ipynb`):
   - Load the trained model
   - Visualize sample predictions
   - Analyze model performance on different digits
   - Create visualizations of model predictions

## Dataset

The MNIST dataset consists of 70,000 handwritten digits (0-9) in grayscale format. Each image is 28x28 pixels. The dataset is automatically downloaded when running the code.

## Tasks

### Part 1: Model Implementation

1. Implement `run_experiment_mnist.py` using `run_experiment_moons.py` as a reference:
   - Load and preprocess the MNIST dataset
   - Implement appropriate data transformations
   - Define a suitable neural network architecture
   - Implement the training loop
   - Track training and validation metrics
   - Save the trained model

### Part 2: Visualization and Analysis

Complete the `visualization_mnist.ipynb` notebook to:
1. Load the trained model
2. Select and display sample images from the test set
3. Show model predictions for these samples
4. Visualize the model's confidence in its predictions
5. Analyze cases where the model makes mistakes
6. Create a confusion matrix to show overall performance

## Evaluation Criteria

Your submission will be evaluated based on:
1. Correct implementation of the MNIST dataset loading and preprocessing
2. Appropriate model architecture and training process
3. Quality of the visualization notebook
4. Analysis of model performance
5. Code organization and documentation

## Submission

Submit your completed:
1. `run_experiment_mnist.py` with your implementation
2. Completed `visualization_mnist.ipynb`

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset Documentation](https://pytorch.org/vision/stable/datasets.html#mnist)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)