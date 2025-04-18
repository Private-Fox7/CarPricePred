
# Car Price Prediction Model

This project implements a Random Forest Regression model to predict car prices based on various features like make, model, technical specifications, and more.

## Features
- Handles both numerical and categorical data using Label Encoding
- Performs hyperparameter tuning for optimal max_leaf_nodes
- Includes visualization of:
  - Actual vs Predicted prices
  - Feature importance analysis
- Provides sample prediction functionality

## Requirements
```
pandas
scikit-learn
matplotlib
numpy
```

## Installation
```bash
pip install pandas sklearn matplotlib numpy
```

## Usage
1. Place your dataset 'CarPrice_Assignment.csv' in the correct path
2. Run the script:
```bash
python CarPricePred.py
```

## Model Details
- Algorithm: Random Forest Regressor
- Evaluation Metric: Mean Absolute Error (MAE)
- Train/Test Split: 80/20

## Results
The model outputs:
- Best tree size for maximum leaf nodes
- Mean Absolute Error
- Predicted price for sample car
- Visualizations of predictions and feature importance

## Dataset
The model uses the Car Price Assignment dataset containing various automobile features and their corresponding prices.

## Author
Private-Fox7

## License
MIT License - See LICENSE file for details
