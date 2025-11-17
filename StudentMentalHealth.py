import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QTextEdit, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate


class MentalHealthGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mental Health Analysis - Naive Bayes | KNN | Decision Tree")
        self.resize(980, 750)

        self.apply_dark_theme()

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 20, 30, 20)

        # Title
        self.label_title = QLabel("🧠 Mental Health Prediction System")
        self.label_title.setFont(QFont("Arial", 22, QFont.Bold))
        self.label_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_title)

        # Dataset label (inside a styled frame)
        self.dataset_frame = QFrame()
        self.dataset_frame.setFrameShape(QFrame.StyledPanel)
        self.dataset_frame.setStyleSheet("background-color: #2b2b2b; padding: 10px; border-radius: 8px;")

        frame_layout = QVBoxLayout()
        self.label_dataset = QLabel("Dataset: Belum dipilih")
        self.label_dataset.setFont(QFont("Arial", 11))
        frame_layout.addWidget(self.label_dataset)
        self.dataset_frame.setLayout(frame_layout)

        layout.addWidget(self.dataset_frame)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        self.btn_load = QPushButton("📂 Load Dataset")
        self.style_button(self.btn_load)
        self.btn_load.clicked.connect(self.load_dataset)
        btn_layout.addWidget(self.btn_load)

        self.btn_run = QPushButton("▶ Run Analysis")
        self.style_button(self.btn_run)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_analysis)
        btn_layout.addWidget(self.btn_run)

        layout.addLayout(btn_layout)

        # Result boxed area
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setStyleSheet("""
            QTextEdit {
                background-color: #1f1f1f;
                color: #ffffff;
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.result_box)

        self.setLayout(layout)
        self.dataset = None

    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#121212"))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor("#1E1E1E"))
        palette.setColor(QPalette.AlternateBase, QColor("#2E2E2E"))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor("#2E2E2E"))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor("#00ADB5"))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

    def style_button(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: #00ADB5;
                color: white;
                padding: 10px 18px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #06c2ca;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
        """)
        button.setCursor(Qt.PointingHandCursor)

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Dataset CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.dataset = pd.read_csv(file_path)
            self.label_dataset.setText(f"Dataset: {file_path}")
            self.btn_run.setEnabled(True)

    def run_analysis(self):
        data = self.dataset[
            ['Choose your gender', 'Do you have Panic attack?',
             'Did you seek any specialist for a treatment?']
        ].dropna()

        # Encoding
        label_encoders = {col: LabelEncoder() for col in data.columns}
        for col in data.columns:
            data[col] = label_encoders[col].fit_transform(data[col])

        X = data.drop('Did you seek any specialist for a treatment?', axis=1)
        y = data['Did you seek any specialist for a treatment?']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        result_text = ""

        # --- Naive Bayes ---
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        pred_nb = nb.predict(X_test)
        acc_nb = accuracy_score(y_test, pred_nb)

        result_text += f"\n=== Naive Bayes ===\nAccuracy: {acc_nb:.2f}\n"
        result_text += tabulate(
            self.class_report_dict(y_test, pred_nb, label_encoders),
            headers=["Class", "Precision", "Recall", "F1", "Support"],
            tablefmt="grid"
        )
        result_text += "\n" + tabulate(confusion_matrix(y_test, pred_nb), tablefmt="grid")

        # --- KNN ---
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, pred_knn)

        result_text += f"\n\n=== KNN ===\nAccuracy: {acc_knn:.2f}\n"
        result_text += tabulate(
            self.class_report_dict(y_test, pred_knn, label_encoders),
            headers=["Class", "Precision", "Recall", "F1", "Support"],
            tablefmt="grid"
        )
        result_text += "\n" + tabulate(confusion_matrix(y_test, pred_knn), tablefmt="grid")

        # --- Decision Tree ---
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        pred_dt = dt.predict(X_test)
        acc_dt = accuracy_score(y_test, pred_dt)

        result_text += f"\n\n=== Decision Tree ===\nAccuracy: {acc_dt:.2f}\n"
        result_text += tabulate(
            self.class_report_dict(y_test, pred_dt, label_encoders),
            headers=["Class", "Precision", "Recall", "F1", "Support"],
            tablefmt="grid"
        )
        result_text += "\n" + tabulate(confusion_matrix(y_test, pred_dt), tablefmt="grid")

        self.result_box.setText(result_text)

    def class_report_dict(self, y_true, y_pred, encoders):
        report = classification_report(
            y_true, y_pred,
            target_names=encoders['Did you seek any specialist for a treatment?'].classes_,
            output_dict=True
        )
        return [
            [cls, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1-score']:.2f}", v['support']]
            for cls, v in report.items() if cls not in ["accuracy"]
        ]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MentalHealthGUI()
    window.show()
    sys.exit(app.exec_())
