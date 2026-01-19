# Brain Tumor Classification

This repository contains a Machine Learning project implemented primarily in MATLAB for the classification of brain tumors. It utilizes Histogram of Oriented Gradients (HOG) for feature extraction and a custom optimization implementation using the Memoryless BFGS algorithm.

## Project Structure

* **`prep_data.py`**: A Python script used to download and prepare the dataset for the MATLAB model.
* **`final_project.m`**: The main driver script to run the classification project.
* **`hog20.m`**: MATLAB function to extract Histogram of Oriented Gradients (HOG) features from the images.
* **`SRMCC_bfgsML.m`**: Implementation of the SRMCC algorithm using BFGS (Broyden–Fletcher–Goldfarb–Shanno) optimization.
* **`bt_lsearch2019.m`**: Implementation of Backtracking Line Search used during optimization.
* **`f_SRMCC.m`**: Calculates the objective function value for the model.
* **`g_SRMCC.m`**: Calculates the gradient vector for the model.
* **`tune_hyperparameter.m`**: Script for tuning model hyperparameters (likely regularization constants).
* **`mu_plot.png`**: Visualization of the hyperparameter tuning results (e.g., accuracy vs. mu).

## Getting Started

### Prerequisites

To run this project, you will need:
1.  **MATLAB** (with Image Processing Toolbox recommended).
2.  **Python 3.x** (for data downloading).

### 1. Dataset Preparation

Before running the MATLAB scripts, you must download the dataset. The project includes a Python helper script for this purpose.

Run the following command in your terminal:

```bash
python prep_data.py
```

This script will download the necessary image data, pre-process the images, and convert them to grayscale floats.

### 2. Running the Classification
Once the data is prepared:
1. Open MATLAB
2. Navigate to the project directory
3. Run the main script: **final_project.m**

This project uses **Histogram of Oriented Gradients** features (**HOG**) to represent the brain tumor images. Hyperparameters were carefully selected through a series of cross-validation. The  model is trained using the Memoryless BFGS algorithm (**`SRMCC_bfgsML.m`**) combined with Backtracking Line Search (**`bt_lsearch2019.m`**) to minimize the objective function.

## Results
The model achieved a test accuracy of **89.47%**, confirming that HOG features successfully captured the edge information and structural characteristics of the tumors. Though this accuracy falls short of Convolutional Neural Networks (CNNs), which typically achieve an accuracy rate between 91% and 97%, this model took less than 15 minutes to train (once the final set of hyperparameters are selected) compared to days to weeks of training for CNNs.

## Credits
**Dataset:** Brain Tumor MRI Dataset from Kaggle (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

**`hog20.m`**, **`SRMCC_bfgsML.m`**, and **`bt_lsearch2019.m`** scripts were provided by the course instructors, with the corresponding authors stated at the top of each script.

