from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load & preprocess data
df = preprocess("Customer-Churn.CSV")

# Split X and y
y = df['churn']
X = df.drop(columns=['churn'])

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", acc)
print(classification_report(y_val, y_pred))

# Save model
joblib.dump(model, "churn_model.pkl")
print("Model saved as churn_model.pkl")
