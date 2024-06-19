import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load the dataset
data = pd.read_csv('Bowlers.csv')

# Preprocess the data
# Encode the 'Training module' column
le = LabelEncoder()
data['Training module'] = le.fit_transform(data['Training module'])

# Features and target variable
X = data.drop(columns=['Training module'])
y = data['Training module']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)


report = classification_report(y_test, y_pred, target_names=le.classes_)
print(report)

# # Save the model to a pickle file
model_filename = 'training_model1.pkl'
with open(model_filename, 'wb') as file:
     pickle.dump((model, le), file)

# Load the model and make a prediction
with open('training_model1.pkl', 'rb') as file:
    loaded_model, loaded_le = pickle.load(file)

# Ensure new data has correct feature names
new_data = pd.DataFrame(np.array([[9,	77.3,	426,	10,	5.5	,42.6,	46.5,	0	,0,	46	,10,	266]]), columns=X.columns)
new_predictions = loaded_model.predict(new_data)
new_predictions_decoded = loaded_le.inverse_transform(new_predictions)
print("New Predictions:", new_predictions_decoded)
