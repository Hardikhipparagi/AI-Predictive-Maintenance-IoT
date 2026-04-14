import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("data/iot_sensor_data.csv")

# Features & target
X = data[['temperature', 'vibration', 'current']]
y = data['failure']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/model.pkl")

# Visualization
plt.scatter(data['temperature'], data['failure'])
plt.xlabel("Temperature")
plt.ylabel("Failure")
plt.savefig("outputs/temp_vs_failure.png")
plt.show()