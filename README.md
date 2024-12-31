
# Multi-Layer Perceptron for Binary Classification

This repository demonstrates the implementation of a Multi-Layer Perceptron (MLP) model using PyTorch for binary classification tasks. It includes data preprocessing, visualization, model training, and evaluation techniques, along with experimental methods for hyperparameter tuning, weight initialization, and threshold-based performance evaluation.



## Features

### 1. **Data Preprocessing**
- Load and preprocess datasets with:
  - Duplicate removal
  - Missing value analysis
  - Feature scaling (standardization, min-max scaling)
  - Feature encoding (one-hot, label, and target encoding)
  - Feature discretization
- Visualizations:
  - Histograms and Q-Q plots for feature distributions
  - Impact of scaling and discretization

### 2. **Exploratory Data Analysis (EDA)**
- Analyze relationships and feature behavior through:
  - Histograms and Q-Q plots
  - Binned feature counts

### 3. **MLP Model for Binary Classification**
- Implements an MLP with:
  - Multiple hidden layers
  - Batch normalization and dropout
  - Flexible activation functions (ReLU, Leaky ReLU, Softmax, Swish)
  - Custom weight initialization techniques (Glorot, He, Lecun, etc.)
- Trains with:
  - Binary Cross-Entropy Loss
  - Adam and SGD optimizers
  - Learning rate scheduling

### 4. **Experimental Techniques**
- **Weight Initialization**: Test multiple initialization methods to improve training convergence.
- **Activation Functions**: Explore activation functions to optimize model performance.
- **Hyperparameter Tuning**: Experiment with the number of layers, neurons, optimizers, and dropout rates.

### 5. **Evaluation and Threshold Analysis**
- Performance metrics:
  - Accuracy, Precision, Recall, F1 Score
  - ROC-AUC and ROC Curve visualization
- Threshold-based analysis:
  - Evaluate metrics across different classification thresholds
  - Visualize trends for better decision-making.



## Results Summary

1. **Training Configurations**:
   - **Activation Functions**: ReLU, Leaky ReLU, Softmax, Swish
   - **Weight Initialization**: Glorot, He, Lecun, Normal, Zero
   - **Optimizer**: Adam and SGD
   - **Hyperparameters**: Hidden layers, neurons, learning rate, dropout

2. **Evaluation**:
   - Performance metrics at different thresholds
   - ROC-AUC and confusion matrix for model evaluation



## Requirements

- Python 3.8+
- Required libraries (install via pip):
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn torch
  ```



## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/MLP-Binary-Classification.git
   cd MLP-Binary-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the code:
   ```bash
   python mlp_binary_classification.py
   ```



## File Structure

- **mlp_binary_classification.py**: Main script containing all steps of the project.
- **Employee.csv**: Example dataset for preprocessing and training.
- **README.md**: Documentation for the repository.



## Dataset

The example dataset (`Employee.csv`) is a binary classification dataset with features like employee performance, tenure, and satisfaction to predict if an employee will leave the organization (`LeaveOrNot`).



## Usage

1. **Preprocessing**:
   - Load your dataset by specifying the file path.
   - Choose preprocessing techniques (scaling, encoding, discretization).

2. **Training the MLP**:
   - Configure hyperparameters such as activation functions, optimizers, and the number of layers.
   - Train the MLP model using `train_improved_model`.

3. **Evaluate the Model**:
   - Use `train_and_evaluate_model` to evaluate metrics like accuracy, precision, and recall.
   - Perform threshold-based analysis for detailed performance insights.

4. **Experiment with Configurations**:
   - Test weight initialization methods.
   - Try different activation functions and hyperparameters to optimize performance.



## Example Visualizations

### Q-Q Plot and Histogram for Feature Distribution
- Analyze normality and feature transformations.

### Threshold-Based Metric Evaluation
- Visualize metrics (accuracy, precision, recall, etc.) across thresholds for better decision-making.

### ROC-AUC Curve
- Assess model performance and compare against random classifiers.



## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.


