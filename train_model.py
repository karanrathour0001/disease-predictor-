import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("dataset.csv")

print("Dataset Loaded ✅")

# -------------------------------
# Clean Column Names (IMPORTANT)
# -------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^\w_]", "", regex=True)
)

print("Columns cleaned ✅")

# -------------------------------
# Features & Target
# -------------------------------
# first column = diseases
X = df.drop("diseases", axis=1)
y = df["diseases"]

# -------------------------------
# Encode Target
# -------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------


model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# -------------------------------
# Accuracy
# -------------------------------
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# Save Files
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("Model saved successfully ✅")