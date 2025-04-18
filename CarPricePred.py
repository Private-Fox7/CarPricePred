import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
carload = 'https://github.com/Private-Fox7/CarPricePred/blob/main/CarPrice_Assignment.csv'
# Read the dataset
car_data = pd.read_csv(carload)

# Display the first 5 rows of the dataset and the shape of the dataset also the describe
print(car_data.head())
print(car_data.describe())

# Selecting the features and the target variable
X = car_data.drop(['car_ID', 'price'], axis=1)
y = car_data['price']

# Encoding categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Convert all data to numeric type
X = X.astype(float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to calculate MAE for a given value of max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=42)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    return mean_absolute_error(val_y, predictions)

# Candidate max_leaf_nodes values to test
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Loop through candidate values of max_leaf_nodes and compute MAE
scores = {leaf_size: get_mae(leaf_size, X_train, X_test, y_train, y_test) for leaf_size in candidate_max_leaf_nodes}

# Store the best value of max_leaf_nodes (it will be one of 5, 25, 50, 100, 250, or 500)
best_tree_size = min(scores, key=scores.get)
print(f"The best tree size (max_leaf_nodes) is -> {best_tree_size}")

# Train the Random Forest model with the best max_leaf_nodes value
best_model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=42)
best_model.fit(X_train, y_train)

# Predict the price of the car
predictval = best_model.predict(X_test)
mae = mean_absolute_error(predictval, y_test)
print(f"Mean Absolute Error -> {mae}")

# Sample prediction
sample = [3, 'audi 5000s (diesel)', 'gas', 'std', 'two', 'hatchback', 'rwd', 'front', 88.6, 168.8, 64.1, 48.8, 2548, 'dohc', 'five', 130, 'mpfi', 3.47, 2.68, 9, 111, 5000, 21, 27]
sample = pd.DataFrame([sample], columns=X.columns)

# Encode categorical columns in the sample
for column in categorical_columns:
    sample[column] = label_encoders[column].transform(sample[column])

# Convert all data to numeric type
sample = sample.astype(float)

# Predict the price for the sample car
predicted_price = best_model.predict(sample)
predicted_price = round(predicted_price[0], 2)
print(f"Predicted price for the sample car: {predicted_price} in dollars")


# Plotting the predicted values vs the actual values
plt.scatter(y_test, predictval)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.show()

# Plotting the feature importance
importances = best_model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.xlabel('Features', color='green')
plt.ylabel('Importance', color='green')
plt.tight_layout()
plt.legend(['Feature Importance'])
plt.show()

