import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from streamlit import session_state

# Initialize session state
def init_session_state():
    if "user_email" not in session_state:
        session_state.user_email = None

# Scatterplot data placeholder
scatterplot_data = None

# Main page
def main():
    init_session_state()

    st.title("SoftGrow")
    st.write("Your data visualization dashboard")

    st.sidebar.header("User Input")
    data_file = st.sidebar.file_uploader("Upload Data File (CSV, Excel)", type=["csv", "xlsx"])

    if data_file is not None:
        st.sidebar.markdown(f"**File Uploaded:** {data_file.name}")

        try:
            global scatterplot_data
            scatterplot_data = process_data_file(data_file)

            # Data Exploration
            st.subheader("Data Exploration")
            st.write(scatterplot_data.describe())

            # Scatterplot
            st.subheader("Scatterplot")
            scatterplot_fig = generate_scatterplot(scatterplot_data)
            st.plotly_chart(scatterplot_fig)

            # Correlation/Counts Visualization
            st.subheader("Correlation/Counts Visualization")
            for column in scatterplot_data.columns:
                if pd.api.types.is_numeric_dtype(scatterplot_data[column]):
                    st.write(f"**{column} (Numeric)**")
                    heatmap_fig = generate_heatmap(scatterplot_data)
                    st.plotly_chart(heatmap_fig)
                else:
                    st.write(f"**{column} (Non-Numeric)**")
                    bar_chart_fig = generate_bar_chart(scatterplot_data, column)
                    st.plotly_chart(bar_chart_fig)

            # Pair Plot
            st.subheader("Pair Plot")
            pair_plot_fig = px.scatter_matrix(scatterplot_data)
            st.plotly_chart(pair_plot_fig)

            # Histograms
            st.subheader("Histograms")
            for column in scatterplot_data.columns:
                histogram_fig = px.histogram(scatterplot_data, x=column, title=f"Histogram of {column}")
                st.plotly_chart(histogram_fig)

            # 3D Scatter Plot
            if len(scatterplot_data.columns) >= 3:
                st.subheader("3D Scatter Plot")
                scatterplot_3d_fig = px.scatter_3d(
                    scatterplot_data,
                    x=scatterplot_data.columns[0],
                    y=scatterplot_data.columns[1],
                    z=scatterplot_data.columns[2]
                )
                st.plotly_chart(scatterplot_3d_fig)

            # Time Series Plot (Showing Trends)
            st.subheader("Time Series Plot")
            time_series_fig = generate_time_series_plot(scatterplot_data)
            st.plotly_chart(time_series_fig)

            # Box Plot (Numerical vs. Categorical)
            st.subheader("Box Plot")
            box_plot_fig = generate_box_plot(scatterplot_data)
            st.plotly_chart(box_plot_fig)

            # Linear Regression
            st.subheader("Linear Regression")
            linear_regression_fig = generate_linear_regression_plot(scatterplot_data)
            st.plotly_chart(linear_regression_fig)

            # Interactive Controls
            st.sidebar.subheader("Interactive Controls")
            selected_columns = st.sidebar.multiselect("Select Columns for X and Y Axes", scatterplot_data.columns)
            if selected_columns:
                st.subheader("Custom Scatterplot")
                custom_scatterplot_fig = generate_scatterplot(scatterplot_data[selected_columns])
                st.plotly_chart(custom_scatterplot_fig)

            st.markdown("Proceed to Payment")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Process data file and return DataFrame
def process_data_file(data_file):
    if data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(data_file)
    elif data_file.type == "text/csv":
        df = pd.read_csv(data_file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    return df

# Generate scatterplot using Plotly
def generate_scatterplot(data, color_column=None, size_column=None):
    fig = px.scatter(data, x=data.columns[0], y=data.columns[1], color=color_column, size=size_column)
    return fig

# Generate heatmap for numeric data
def generate_heatmap(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_columns].corr()
    fig = px.imshow(correlation_matrix, color_continuous_scale="Viridis", title="Correlation Heatmap")
    return fig

# Generate bar chart for non-numeric data
def generate_bar_chart(data, column):
    counts = data[column].value_counts()
    fig = px.bar(x=counts.index, y=counts.values, labels={'x': column, 'y': 'Count'}, title=f"Counts of {column}")
    return fig

# Generate time series plot using Plotly
def generate_time_series_plot(data):
    fig = px.line(data, x=data.columns[0], y=data.columns[1], labels={data.columns[0]: "Time", data.columns[1]: "Value"})
    return fig

# Generate box plot
def generate_box_plot(data):
    fig = px.box(data, x=data.columns[2], y=data.columns[1], title="Box Plot")
    return fig

# Generate linear regression plot
def generate_linear_regression_plot(data):
    selected_columns = [data.columns[0], data.columns[1]]

    if data[selected_columns].isnull().any().any():
        imputer = SimpleImputer(strategy="mean")
        data[selected_columns] = imputer.fit_transform(data[selected_columns])

    X = data[selected_columns[0]].values.reshape(-1, 1)
    y = data[selected_columns[1]].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    regression_line = go.Scatter(x=X_test.flatten(), y=y_pred.flatten(), mode="lines", name="Regression Line")

    fig = go.Figure(data=[go.Scatter(x=X.flatten(), y=y.flatten(), mode="markers", name="Data Points"), regression_line])
    fig.update_layout(title=f"Linear Regression (MSE: {mse:.2f})", xaxis_title=selected_columns[0], yaxis_title=selected_columns[1])

    return fig

if __name__ == "__main__":
    main()

