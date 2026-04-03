#step 1,2,3, done
import pandas as pd
df = pd.read_csv("iris.csv")

#step 4, explore dataset
print(df.head(10))
print("Before cleaning:", df.shape)
df.info()
print(df.describe())

#step 5, data cleaning
print(df.isnull().sum()) # Check for missing values in each column
df = df.dropna()  # Remove rows with missing values
print("After cleaning:", df.shape)

#step 6, encode categorical variables
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species']) #conerting text to num
print(df.head())

#step 7, split features and target
X = df.drop('species', axis=1)
y = df['species']
print(X.head())
print(y.head())
print(X.shape)
print(y.shape)

#step 8, train test split
from sklearn.model_selection import train_test_split #train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

#step 9, feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[:5])

#step 10, build neural network model using keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Create model
model = Sequential()
# Input + Hidden Layer 1
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
# Hidden Layer 2
model.add(Dense(8, activation='relu'))
# Output Layer (3 classes)
model.add(Dense(3, activation='softmax'))
# Show model structure
model.summary()

#step 11, compile model 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=50) #step 12, train model

#step 13, evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

#step 14, predictions
import numpy as np
# Get prediction probabilities
predictions = model.predict(X_test)
# Convert probabilities → class labels
y_pred = np.argmax(predictions, axis=1)
print("Predicted classes:", y_pred[:10])
print("Actual classes:   ", y_test.values[:10])