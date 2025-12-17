import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


st.set_page_config(page_title="Customer Churn ANN", layout="wide")
st.title("📉 Customer Churn Prediction using ANN")

st.markdown("""
This app trains an **Artificial Neural Network (ANN)** to predict customer churn  
using the **Churn_Modelling dataset**.
""")

df = pd.read_csv("customer_churn_dataset-testing-master.csv")  

st.subheader("Raw Dataset")
st.dataframe(df.head())

df.drop(['CustomerID'], axis=1, inplace=True)
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.sidebar.header("⚙️ Hyperparameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64])
epochs = st.sidebar.slider("Epochs", 10, 100, 50)

if st.sidebar.button("🚀 Train Model"):
    with st.spinner("Training ANN..."):

        model = Sequential()
        model.add(Dense(units=6, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(units=6, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        optimizer = SGD(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=0
        )

    st.success("✅ Model training completed")

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5)

    cm = confusion_matrix(y_test, y_pred_binary)
    accuracy = (cm[0, 0] + cm[1, 1]) / len(y_test) * 100

    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}%")
        st.text("Classification Report")
        st.text(
            classification_report(
                y_test,
                y_pred_binary,
                target_names=['Not Churn', 'Churn']
            )
        )

    with col2:
        fig, ax = plt.subplots()
        im = ax.matshow(cm, cmap='Blues')
        plt.colorbar(im)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Not Churn', 'Churn'])
        ax.set_yticklabels(['Not Churn', 'Churn'])
        plt.title("Confusion Matrix")
        st.pyplot(fig)

    st.subheader("📈 Training History")

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))

    ax2[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax2[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2[0].set_title("Accuracy")
    ax2[0].set_xlabel("Epochs")
    ax2[0].legend()

    ax2[1].plot(history.history['loss'], label='Train Loss')
    ax2[1].plot(history.history['val_loss'], label='Validation Loss')
    ax2[1].set_title("Loss")
    ax2[1].set_xlabel("Epochs")
    ax2[1].legend()

    st.pyplot(fig2)

    previous_accuracy = 85
    st.subheader("📌 Accuracy Comparison")
    st.write(f"Previous ANN Accuracy: **{previous_accuracy:.2f}%**")
    st.write(f"Current ANN Accuracy: **{accuracy:.2f}%**")
    st.write(f"Difference: **{accuracy - previous_accuracy:.2f}%**")
