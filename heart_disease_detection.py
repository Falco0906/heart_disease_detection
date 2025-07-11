import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load dataset
df = pd.read_csv("dataset.csv")

# Create models directory
os.makedirs("models", exist_ok=True)

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if model and scaler exist
model_path = "models/random_forest.pkl"
scaler_path = "models/scaler.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("✅ Loading existing model and scaler...")
    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

else:
    print("⚙️ No saved model found. Training new models...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True)
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        print(f"\n====== {name} ======")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            print("ROC-AUC Score:", auc)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    best_model = models["Random Forest"]
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print("💾 Model and scaler saved.")

# ======== Predict on New Patients in a Loop ========
print("\n--- Heart Disease Prediction ---")
print("Enter patient values as a comma-separated list in this order:")
print("Age, Sex (0=female,1=male), Chest Pain (1-4), Resting BP, Cholesterol, FBS (0/1), ECG (0-2), Max HR, Exercise Angina (0/1), Oldpeak, ST Slope (1-3)")
print("Type 'exit' to quit.\n")

while True:
    input_string = input("Enter values here: ")

    if input_string.strip().lower() == "exit":
        print("👋 Exiting prediction loop.")
        break

    try:
        input_list = [float(x.strip()) for x in input_string.split(",")]
        if len(input_list) != 11:
            raise ValueError("❌ You must enter exactly 11 values.")
        new_patient_scaled = scaler.transform([input_list])
        prediction = best_model.predict(new_patient_scaled)

        print("\n====== Prediction Result ======")
        if prediction[0] == 1:
            print("⚠️ The model predicts: The patient may have heart disease.")
        else:
            print("✅ The model predicts: The patient is likely normal.")
        print("\n🔁 Enter another patient's data or type 'exit' to quit.\n")

    except Exception as e:
        print("❌ Error:", e)
        print("Please enter values correctly.")
