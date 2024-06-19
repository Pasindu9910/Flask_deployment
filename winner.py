import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np

import sklearn
print("scikit-learn version:", sklearn.__version__)

# Load the data
data = pd.read_csv('match.csv')

# Define features and target variable
X = data[['Team1', 'Team2', 'innings1_overs', 'innings1_wickets', 'innings1_runs']].astype(float)
y = data['Winner']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Save the model
with open('winning.pkl', 'wb') as file:
    pickle.dump(model, file)

# Ensure you load the model in the same environment and version of scikit-learn
with open('winning.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

new_data2 = np.array([[14.0, 6.0, 50.0, 7.0, 310.0]]).astype(float)
# Make predictions on new data
new_predictions = loaded_model.predict(new_data2)
print("New Predictions:", new_predictions)

