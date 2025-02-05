import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate

# Load dataset
file_path = 'Student Mental Health Dataset.csv'
dataset = pd.read_csv(file_path)

print("Nama Kolom dalam Dataset:")
print(dataset.columns)

columns = [
    'Choose your gender', 'Do you have Panic attack?', 
    'Did you seek any specialist for a treatment?'
]
data = dataset[columns].dropna()

# Encode categorical variables
label_encoders = {col: LabelEncoder() for col in data.columns}
for col in data.columns:
    data[col] = label_encoders[col].fit_transform(data[col])

# Split data into features and target
X = data.drop('Did you seek any specialist for a treatment?', axis=1)
y = data['Did you seek any specialist for a treatment?']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes (GaussianNB)
print("\n=== Naive Bayes (GaussianNB) ===")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy: {accuracy_nb:.2f}")
report_nb = classification_report(
    y_test, y_pred_nb, 
    target_names=label_encoders['Did you seek any specialist for a treatment?'].classes_,
    output_dict=True
)
report_table_nb = [
    [key, f"{values['precision']:.2f}", f"{values['recall']:.2f}", f"{values['f1-score']:.2f}", values['support']]
    for key, values in report_nb.items() if key not in ['accuracy']
]
print("\nClassification Report:")
print(tabulate(report_table_nb, headers=["Class", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))
cm_nb = confusion_matrix(y_test, y_pred_nb)
print("\nConfusion Matrix:")
print(tabulate(cm_nb, headers=["Predicted No", "Predicted Yes"], tablefmt="grid"))

# k-Nearest Neighbors (KNN)
print("\n=== k-Nearest Neighbors (KNN) ===")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy: {accuracy_knn:.2f}")

# Classification report with zero_division=1 to avoid warning
report_knn = classification_report(
    y_test, y_pred_knn, 
    target_names=label_encoders['Did you seek any specialist for a treatment?'].classes_,
    output_dict=True,
    zero_division=1
)
report_table_knn = [
    [key, f"{values['precision']:.2f}", f"{values['recall']:.2f}", f"{values['f1-score']:.2f}", values['support']]
    for key, values in report_knn.items() if key not in ['accuracy']
]
print("\nClassification Report:")
print(tabulate(report_table_knn, headers=["Class", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))

# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\nConfusion Matrix:")
print(tabulate(cm_knn, headers=["Predicted No", "Predicted Yes"], tablefmt="grid"))

# Decision Tree
print("\n=== Decision Tree ===")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy: {accuracy_dt:.2f}")
report_dt = classification_report(
    y_test, y_pred_dt, 
    target_names=label_encoders['Did you seek any specialist for a treatment?'].classes_,
    output_dict=True,
    zero_division=0
)
report_table_dt = [
    [key, f"{values['precision']:.2f}", f"{values['recall']:.2f}", f"{values['f1-score']:.2f}", values['support']]
    for key, values in report_dt.items() if key not in ['accuracy']
]
print("\nClassification Report:")
print(tabulate(report_table_dt, headers=["Class", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("\nConfusion Matrix:")
print(tabulate(cm_dt, headers=["Predicted No", "Predicted Yes"], tablefmt="grid"))

# Summary
print("\n=== Model Comparison ===")
comparison_table = [
    ["GaussianNB", accuracy_nb],
    ["KNN", accuracy_knn],
    ["Decision Tree", accuracy_dt]
]
print(tabulate(comparison_table, headers=["Model", "Accuracy"], tablefmt="grid"))

