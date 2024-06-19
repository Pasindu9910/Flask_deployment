import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load the dataset
data = pd.read_csv('crick1.csv')

# Preprocess the data
# Encode the 'Training model' column (corrected typo)
le = LabelEncoder()
data['Training modle'] = le.fit_transform(data['Training modle'])

# Features and target variable
X = data.drop(columns=['Training modle'])
y = data['Training modle']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=le.classes_)
print(report)

# Save the model to a pickle file
model_filename = 'training_model.pkl'
with open(model_filename, 'wb') as file:
     pickle.dump((model, le), file)

# Load the model and label encoder from the pickle file
with open('training_model.pkl', 'rb') as file:
    loaded_model, loaded_le = pickle.load(file)

# New data for prediction
new_data = np.array([[164,167,27.33,98.2,54,11,10,1,0]])
new_predictions = loaded_model.predict(new_data)
new_predictions_decoded = loaded_le.inverse_transform(new_predictions)
print("New Predictions:", new_predictions_decoded)
