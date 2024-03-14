import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import plotly.express as px

# Streamlit app
st.title('Online Payment Fraud Detection')
st.markdown("---")

# Center-align button
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    if st.button('Predict', key='predict_button', help="Click to predict fraud"):
        st.subheader('Predicting Fraud')
        st.write("Analyzing data...")

        # Load the dataset
        data = pd.read_csv("PS_20174392719_1491204439457_log.csv")

        # Map categorical values
        data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                         "CASH_IN": 3, "TRANSFER": 4,
                                         "DEBIT": 5})
        data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

        # Split the data
        x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
        y = np.array(data["isFraud"])

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(xtrain, ytrain)

        # Display graph
        st.subheader('Distribution of Transaction Type')
        type_counts = data["type"].value_counts()
        fig = px.pie(type_counts, values=type_counts.values, names=type_counts.index, hole=0.5)
        st.plotly_chart(fig)

        # Prediction of a random value
        random_index = np.random.randint(len(xtest))
        random_feature = xtest[random_index].reshape(1, -1)
        feature_names = ["Transaction Type", "Amount", "Old Balance", "New Balance"]

        st.subheader('Attributes of Random Variable')
        attr_col1, attr_col2 = st.columns(2)
        for i, feature_name in enumerate(feature_names):
            attr_col1.write(f"**{feature_name}:**")
            attr_col2.write(str(random_feature[0][i]))

        prediction = model.predict(random_feature)
        st.subheader('Prediction Result')
        if prediction == "No Fraud":
            st.write("## **No Fraud**")
        else:
            st.write("## **Fraud**")

        # Calculate metrics
        y_pred = model.predict(xtest)
        accuracy = accuracy_score(ytest, y_pred)
        precision = precision_score(ytest, y_pred, pos_label="Fraud")
        loss = model.score(xtest, ytest)
        confusion = confusion_matrix(ytest, y_pred)

        st.subheader('Model Evaluation Metrics')
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Loss: {loss}")
        st.write("Confusion Matrix:")
        st.write(confusion)
        st.write("Classification Report:")
        st.write(classification_report(ytest, y_pred))

        st.markdown("---")
        st.write("Disclaimer: This prediction is based on a random sample from the dataset.")
