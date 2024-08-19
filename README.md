# Fraud Detection Using Deep Learning Hybrid Model

## Overview

`Fraud Detection Using Deep Learning Hybrid Model` is a project focused on detecting fraudulent activities using a hybrid deep learning model. The project involves data preprocessing, model building, training, evaluation, and prediction, utilizing advanced techniques to achieve high accuracy in fraud detection.

### Key Features

- **Data Loading and Preprocessing**: Load datasets from CSV files and preprocess them by scaling features and splitting them into training and testing sets.
- **Model Building and Training**: Build and train deep learning models using TensorFlow/Keras with customizable parameters.
- **Model Evaluation**: Evaluate the performance of the model using metrics like accuracy, confusion matrix, and classification reports.
- **Model Prediction**: Make predictions on new data using the trained model.
- **Visualization**: Plot training metrics such as loss and accuracy over epochs to analyze model performance.

Ensure that you have Python 3.6 or higher installed. You can install the necessary dependencies using the following command:
```bash
pip install -r requirements.txt
```
Alternatively, you can manually install the dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib
```

## Usage
### Running the Project
The main code for the project is contained within the Jupyter notebook Fraud_Detection_Hybrid_Deep_Learning_Model.ipynb. You can run this notebook step by step to perform the following tasks:

- Load and preprocess data
- Build and train the model
- Evaluate model performance
- Make predictions on new data
- Running the Jupyter Notebook

To open and run the Jupyter notebook, use the following command:
```bash
jupyter notebook Fraud_Detection_Hybrid_Deep_Learning_Model.ipynb
```

## Command Line Usage
The project also includes modular Python scripts that can be used directly from the command line or integrated into other projects.

### Loading and Preprocessing Data
```bash
from FraudDetectionHybrid.data_processing import load_data, preprocess_data, split_data

# Load data
data = load_data('data/raw/dataset.csv')

# Preprocess data
X, y = preprocess_data(data, target_column='target')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)
```

### Building and Training the Model
```bash
from FraudDetectionHybrid.model import build_model, train_model

# Build the model
model = build_model(input_shape=X_train.shape[1])

# Train the model
history = train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
```

### Evaluating the Model
```bash
from FraudDetectionHybrid.model import evaluate_model

# Evaluate the model
accuracy, confusion_matrix, classification_report = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_matrix}")
print(f"Classification Report:\n{classification_report}")
```

### Making Prediction
```bash
from FraudDetectionHybrid.model import predict

# Predict on new data
X_new = X_test[:5]  # Example new data
predictions = predict(model, X_new)
print(f"Predictions: {predictions}")
```

### Saving and Loading the Model
```bash
from FraudDetectionHybrid.utils import save_model, load_model

# Save the model
save_model(model, 'model_path.h5')

# Load the model
loaded_model = load_model('model_path.h5')
```

### Plotting Training Metrics
```bash
from FraudDetectionHybrid.utils import plot_metrics

# Plot training metrics
plot_metrics(history)
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions
Contributions to this project are welcome. If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

## Contact
For any questions or inquiries, please contact:
- Author: `Manav Gupta`
- Email: `manav26102002@gmail.com`