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