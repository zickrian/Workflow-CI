import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import warnings
import argparse
import os
warnings.filterwarnings("ignore")

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='diabetes_preprosesing.csv')
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

# --- LOAD PREPROCESSED DATA ---
df = pd.read_csv(args.data_path)

# --- SPLIT FEATURES AND TARGET ---
X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
)

# --- TRACKING & TRAINING ---
mlflow.sklearn.autolog()  

with mlflow.start_run(run_name="RandomForest_Diabetes_Baseline") as run:
	# Using class_weight='balanced' to handle imbalanced data without SMOTE
	model = RandomForestClassifier(
		random_state=args.random_state,
		class_weight='balanced',
		n_estimators=100,
		max_depth=10
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	# --- EVALUATION ---
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	# Log metrics explicitly
	mlflow.log_metric("accuracy", accuracy)
	mlflow.log_metric("precision", precision)
	mlflow.log_metric("recall", recall)
	mlflow.log_metric("f1_score", f1)

	# Log the model explicitly
	mlflow.sklearn.log_model(model, "model")

	print("\n" + "="*50)
	print("MODEL EVALUATION RESULTS")
	print("="*50)
	print(f"Accuracy  : {accuracy:.4f}")
	print(f"Precision : {precision:.4f}")
	print(f"Recall    : {recall:.4f}")
	print(f"F1 Score  : {f1:.4f}")
	print("="*50)

	print("\nCLASSIFICATION REPORT:")
	print(classification_report(y_test, y_pred))
	print("="*50 + "\n")

	# Print run_id for easy extraction
	print(f"MLflow Run ID: {run.info.run_id}")
	
	# Save run_id to a file for easy retrieval in CI/CD
	with open("mlflow_run_id.txt", "w") as f:
		f.write(run.info.run_id)
