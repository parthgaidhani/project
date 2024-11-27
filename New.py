import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import openai
import os

# Set OpenAI API key

# App Title
st.title("AI-Powered Data Analysis and Prediction")
st.write("An intelligent dashboard for data insights and predictions.")

# Sidebar for file upload
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Function to read uploaded file
def read_file(file):
    if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(file)
    elif file.type == "text/csv":
        return pd.read_csv(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None

# AI Analysis Function
def ai_analysis(data):
    prompt = (
        "You are a data scientist. Analyze the following dataset and provide insights, patterns, and predictions:\n\n"
        f"{data.head(10).to_json()}\n\n"
        "Include a summary of trends, anomalies, and potential insights."
    )
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Generate scatterplot
def generate_scatterplot(data, x_col, y_col):
    fig = px.scatter(data, x=x_col, y=y_col, title=f"Scatterplot: {x_col} vs {y_col}")
    return fig

# Linear Regression Prediction
def linear_regression_prediction(data, x_col, y_col):
    X = data[x_col].values.reshape(-1, 1)
    y = data[y_col].values.reshape(-1, 1)

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    y = imputer.fit_transform(y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Plot Results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', name='Actual Data'))
    fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_pred.flatten(), mode='lines', name='Predicted Line'))
    fig.update_layout(
        title=f"Linear Regression Prediction (MSE: {mse:.2f})",
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    return fig

# Main App Logic
if uploaded_file:
    # Read data
    data = read_file(uploaded_file)
    if data is not None:
        st.write("### Uploaded Dataset")
        st.write(data.head())

        # Display AI Analysis
        st.write("### AI-Driven Insights")
        if st.button("Run AI Analysis"):
            try:
                insights = ai_analysis(data)
                st.text_area("AI Analysis Results", insights, height=200)
            except Exception as e:
                st.error(f"Error in AI Analysis: {e}")

        # Data Visualization
        st.write("### Data Visualization")
        columns = data.columns.tolist()
        x_axis = st.selectbox("Select X-axis column", columns, index=0)
        y_axis = st.selectbox("Select Y-axis column", columns, index=1)

        if x_axis and y_axis:
            scatter_fig = generate_scatterplot(data, x_axis, y_axis)
            st.plotly_chart(scatter_fig)

        # AI Predictions
        st.write("### AI Prediction: Linear Regression")
        if x_axis and y_axis:
            try:
                regression_fig = linear_regression_prediction(data, x_axis, y_axis)
                st.plotly_chart(regression_fig)
            except Exception as e:
                st.error(f"Error in Linear Regression: {e}")

else:
    st.info("Upload a CSV or Excel file to get started.")
