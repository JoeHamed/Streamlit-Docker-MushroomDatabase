import pandas as pd
import numpy as np
import sklearn
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

def main():
    st.title('Binary Classification Web App')
    st.sidebar.title('Binary Classification')
    st.markdown("Are your Mushrooms Poisonous or Edible? 🍄")
    st.sidebar.markdown("Mushrooms Poisonous or Edible? 🍄")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('./data/mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data(persist=True)
    def split(df): # Create the X and y (Train and Test)
        y = df['class']
        X = df.drop(columns=['class'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            fig_roc = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names) # Calls the predict method internally
            st.pyplot(fig_roc.figure_)

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            fig_roc = RocCurveDisplay.from_estimator(model, X_test, y_test)
            st.pyplot(fig_roc.figure_)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            fig_roc = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
            st.pyplot(fig_roc.figure_)

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    class_names = ['Edible', 'Poisonous']

    # Sidebar layout
    st.sidebar.subheader('Choose the Classifier')
    classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine SVM', 'Logistic Regression', 'Random Forest'))

    if classifier == 'Support Vector Machine SVM':
        st.sidebar.subheader('Choose the Kernel')  # the sidebars are arranged linearly
        kernel = st.sidebar.radio('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'), key='kernel')
        st.sidebar.link_button('Show kernel Documentation', 'https://scikit-learn.org/1.5/modules/svm.html#svm-kernels')

        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step=0.01, key='C')
        gamma = st.sidebar.radio('Gamma (Kernel Coefficient)', ('scale', 'auto'), key='gamma')

        metrics = st.sidebar.multiselect('What Metrics to Plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Support Vector Machine SVM Results')
            model = SVC(kernel=kernel, C=C, gamma=gamma)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write('Accuracy:', round(accuracy, 2))
            st.write('Precision:', round(sklearn.metrics.precision_score(y_test, y_pred, labels=class_names), 2))
            st.write('Recall:', round(sklearn.metrics.recall_score(y_test, y_pred, labels=class_names), 2))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider('Maximum number of iterations', 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect('What Metrics to Plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Logistic Regression Results')
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write('Accuracy:', round(accuracy, 2))
            st.write('Precision:', round(sklearn.metrics.precision_score(y_test, y_pred, labels=class_names), 2))
            st.write('Recall:', round(sklearn.metrics.recall_score(y_test, y_pred, labels=class_names), 2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators = st.sidebar.number_input('Number of estimators (No. of Trees)', 100, 2000, step=5, key='n_estimators')
        max_depth = st.sidebar.number_input("The Maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.checkbox('Bootstrap', value=False) #  instead of training on all the observations, each tree of RF is trained on a subset of the observations.

        metrics = st.sidebar.multiselect('What Metrics to Plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Logistic Regression Results')
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1) # use all the cores for processing
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write('Accuracy:', round(accuracy, 2))
            st.write('Precision:', round(sklearn.metrics.precision_score(y_test, y_pred, labels=class_names), 2))
            st.write('Recall:', round(sklearn.metrics.recall_score(y_test, y_pred, labels=class_names), 2))
            plot_metrics(metrics)


    # Checkboxes and data view
    if st.sidebar.checkbox("Show Raw Dataset", False):  # For Debugging Purposes
        st.subheader('Mushrooms Dataset UCI')
        st.write(df)

if __name__ == '__main__':
    main()
