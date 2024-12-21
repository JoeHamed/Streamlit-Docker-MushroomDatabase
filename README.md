# Binary Classification Web App: Edible vs Poisonous Mushrooms 🍄

This project is a **binary classification web application** built with **Streamlit**. It predicts whether mushrooms are **edible** or **poisonous** based on user-input features. The app supports multiple classifiers, including **Support Vector Machines (SVM)**, **Logistic Regression**, and **Random Forests**. It provides visualization of model performance metrics such as confusion matrix, ROC curve, and precision-recall curve.

---

## Features

- User-friendly **Streamlit interface** for interactive model selection and visualization.
- **Support for multiple classifiers**:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
- **Performance metrics visualization**:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
- Integrated with the **Mushrooms Dataset** from the UCI Machine Learning Repository.
- Fully containerized using **Docker** for seamless deployment.

---

## Project Structure
```bash
. ├── app/
│ ├── app.py # Streamlit app code
│ ├── requirements.txt # Python dependencies
├── data/ │ └── mushrooms.csv # Mushrooms dataset
├── Dockerfile # Docker configuration
└── README.md # Project documentation
```

---

## Dataset

The app uses the **Mushrooms Dataset** from the UCI Machine Learning Repository. Each feature in the dataset has been encoded numerically for compatibility with the classifiers. The target variable is `class`, with two possible values:
- **Edible** (0)
- **Poisonous** (1)

---

## Requirements

- **Docker**: Ensure Docker is installed on your system.
- **Python 3.11** (optional, if running locally without Docker).
- Python dependencies (listed in `requirements.txt`).

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/mushroom-classification-streamlit.git
cd mushroom-classification-streamlit
```

### 2. Build and Run the Docker Container
Build the Docker Image
```bash
docker build -t mushroom-classifier .
```
Run the Docker Container
```bash
docker run -d -p 9000:9000 mushroom-classifier
```
The app will be accessible at `http://localhost:9000`.

## Usage
### Web Application
- Open your browser and navigate to: http://localhost:9000
- Use the sidebar to:
  - Select the classifier (SVM, Logistic Regression, Random Forest).
  - Adjust hyperparameters for the selected model.
  - Choose performance metrics to visualize.
- Upload a new dataset or use the default dataset for predictions.
