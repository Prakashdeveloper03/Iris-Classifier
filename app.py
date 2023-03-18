import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Sets page's title and icon
st.set_page_config(page_title="S&P 500 App", page_icon="🎯")

# Define the app header
st.title("Iris Flower Classifier")

# Define the sidebar header and user input parameters
st.sidebar.header("User Input Parameters")


def user_input_features():
    # Define slider ranges and default values for user input
    sepal_length = st.sidebar.slider("Sepal length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 0.2)
    # Store user input in a dictionary
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    # Convert dictionary to a dataframe and return it
    return pd.DataFrame(data, index=[0])


# Call the user_input_features function to get user input and store it in a dataframe
df = user_input_features()

# Display user input in the app
st.subheader("User Input parameters")
st.write(df)

# Load iris dataset and assign features to X and target to Y
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Predict the flower type and probability of prediction using the trained model
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Display the class labels and their corresponding index number
st.subheader("Class labels and their corresponding index number")
st.write(iris.target_names)

# Display the predicted flower type
st.subheader("Prediction")
st.write(iris.target_names[prediction])

# Display the probability of predicted flower type
st.subheader("Prediction Probability")
st.write(prediction_proba)
