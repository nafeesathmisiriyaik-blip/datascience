import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense


def section(title: str) -> None:
	"""Print a clear section header for readable console output."""
	print("\n" + "=" * 60)
	print(title)
	print("=" * 60)


# 1) Load dataset
df = pd.read_csv("iris.csv")

section("1) First 10 Rows")
print(df.head(10))

section("2) Dataset Info")
df.info()

section("3) Summary Statistics")
print(df.describe())

# 4) Check missing values
section("4) Missing Values Per Column")
print(df.isnull().sum())

# Option 1: remove rows with missing values
df_dropna = df.dropna()
print("Rows/columns after dropna():", df_dropna.shape)

# Option 2: fill missing numeric values with column mean and text with mode
df_filled = df.copy()
numeric_cols = df_filled.select_dtypes(include=["number"]).columns
text_cols = df_filled.select_dtypes(exclude=["number"]).columns

df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
for col in text_cols:
	if not df_filled[col].mode().empty:
		df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])

print("Missing values after fillna handling:")
print(df_filled.isnull().sum())

# 5) Encode target labels (species text -> integer class id)
section("5) Categorical Encoding (LabelEncoder)")
encoder = LabelEncoder()
df_encoded = df_filled.copy()
df_encoded["species_encoded"] = encoder.fit_transform(df_encoded["species"])

print(df_encoded[["species", "species_encoded"]].head(10))
print("Class mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# 6) Split into features (X) and target (y)
section("6) Split Features and Target")
X = df_filled.drop(columns=["species"])
y = df_encoded["species_encoded"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X columns:", list(X.columns))

# 7) Train-test split (80/20)
section("7) Train-Test Split (80/20)")
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y,
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 8) Standardize feature values using training-set statistics only
section("8) Feature Scaling (StandardScaler)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

# 9) Build neural network
section("9) Build Neural Network (Keras)")
model = Sequential(
	[
		Input(shape=(X_train_scaled.shape[1],)),
		Dense(16, activation="relu"),
		Dense(8, activation="relu"),
		Dense(len(encoder.classes_), activation="softmax"),
	]
)

model.summary()

# 10) Compile model
model.compile(
	optimizer="adam",
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"],
)
print("Model compiled successfully.")

# 11) Train model
section("10) Train Model")
history = model.fit(
	X_train_scaled,
	y_train,
	epochs=30,
	batch_size=16,
	validation_data=(X_test_scaled, y_test),
	verbose=1,
)

# 12) Evaluate model on test data
section("11) Evaluate Model")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# 13) Predict class probabilities, then convert to class labels with argmax
section("12) Predictions")
y_prob = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("Predicted class labels (first 10):", y_pred[:10])
print("Actual class labels (first 10):   ", y_test.values[:10])
