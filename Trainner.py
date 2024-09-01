# This part is responsible for training the model. It loads the numpy arrays, 
# train it, and save the trained model. this part is not required for running the code

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import numpy as np


X = np.load('X.npy')
y = np.load('y.npy')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'factory_noise_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
