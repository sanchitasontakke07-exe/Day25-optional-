import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("student_record.csv")

# Drop StudentID
df = df.drop(columns=["StudentID"])

# Encode categorical columns
encoders = {}

categorical_columns = ["Gender", "Internet", "Extracurricular", "Result"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Features and Target
X = df.drop("Result", axis=1)
y = df["Result"]

print("\nData Types After Encoding:\n")
print(X.dtypes)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))

# Save files
joblib.dump(model, "student_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\nFiles Saved Successfully!")