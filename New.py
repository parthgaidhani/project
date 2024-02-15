import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from streamlit import session_state
from sklearn.impute import SimpleImputer

# Scatterplot data placeholder
scatterplot_data = None


# Main page
def main():
    init_session_state()

    st.title("Advanced Scatterplot Analysis")

    st.sidebar.header("User Input")
    data_file = st.sidebar.file_uploader(
        "Upload Data File (CSV, Excel)"
    )

    if data_file is not None:
        st.sidebar.markdown("File Uploaded Successfully!")
        st.sidebar.markdown(f"File Name: {data_file.name}")
        st.sidebar.markdown(f"File Type: {data_file.type}")

        try:
            # Process the data file
            global scatterplot_data
            scatterplot_data = process_data_file(data_file)

            # Display scatterplot
            st.subheader("Scatterplot")
            scatterplot_fig = generate_scatterplot(scatterplot_data)
            st.plotly_chart(scatterplot_fig)

            # Data Exploration
            st.subheader("Data Exploration")
            st.write(scatterplot_data.describe())

            # Heatmap or Bar chart based on column types
            st.subheader("Correlation/Counts Visualization")

            for column in scatterplot_data.columns:
                if pd.api.types.is_numeric_dtype(scatterplot_data[column]):
                    # Numeric data: Use heatmap
                    st.write(f"**{column} (Numeric)**")
                    heatmap_fig = generate_heatmap(scatterplot_data, column)
                    st.plotly_chart(heatmap_fig)
                else:
                    # Non-numeric data: Use bar chart
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
                histogram_fig = px.histogram(
                    scatterplot_data, x=column, title=f"Histogram of {column}"
                )
                st.plotly_chart(histogram_fig)

            # 3D Scatter Plot
            st.subheader("3D Scatter Plot")
            scatterplot_3d_fig = px.scatter_3d(
                scatterplot_data,
                x=scatterplot_data.columns[0],
                y=scatterplot_data.columns[1],
                z=scatterplot_data.columns[2],
            )
            st.plotly_chart(scatterplot_3d_fig)

            # Time Series Plot (Showing Trends)
            st.subheader("Time Series Plot")
            time_series_fig = generate_time_series_plot(scatterplot_data)
            st.plotly_chart(time_series_fig)

            # Box Plot (Visualizing Numerical vs. Categorical)
            st.subheader("Box Plot (Numerical vs. Categorical)")
            box_plot_fig = generate_box_plot(
                scatterplot_data, scatterplot_data.columns[2]
            )
            st.plotly_chart(box_plot_fig)

            # Linear Regression (Predictive Models and Regression Analysis)
            st.subheader("Linear Regression")
            selected_columns = [
                scatterplot_data.columns[0],
                scatterplot_data.columns[1],
            ]
            linear_regression_fig = generate_linear_regression_plot(
                scatterplot_data, selected_columns
            )
            st.plotly_chart(linear_regression_fig)

            # Interactive Controls
            st.sidebar.subheader("Interactive Controls")
            selected_columns = st.sidebar.multiselect(
                "Select Columns for X and Y Axes", scatterplot_data.columns
            )
            color_column = st.sidebar.selectbox(
                "Select Color Column (Optional)", scatterplot_data.columns, index=0
            )
            size_column = st.sidebar.selectbox(
                "Select Size Column (Optional)", scatterplot_data.columns, index=1
            )
            if selected_columns:
                st.subheader("Custom Scatterplot")
                custom_scatterplot_fig = generate_scatterplot(
                    scatterplot_data[selected_columns], color_column, size_column
                )
                st.plotly_chart(custom_scatterplot_fig)
                
                def process_data_file(data_file):
                    if data_file.type == "application/vnd.ms-excel":
                        df = pd.read_excel(data_file)
                    elif data_file.type == "text/csv":
                        df = pd.read_csv(data_file)
                    else:
                        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
                        
                        return df
                        def generate_scatterplot(data, color_column=None, size_column=None):
                            fig = px.scatter(
                                data,
                                x=data.columns[0],
                                y=data.columns[1],
                                color=color_column,
                                size=size_column,
                                labels={data.columns[0]: "X-axis", data.columns[1]: "Y-axis"},)
                            return fig
                            
                            def generate_heatmap(data, column):
                                # Exclude non-numeric columns
                                numeric_columns = data.columns[data.dtypes.apply(lambda c: pd.api.types.is_numeric_dtype(c))]
                                correlation_matrix = data[numeric_columns].corr()
                                fig = px.imshow(
                                    correlation_matrix,
                                    labels=dict(color="Correlation"),
                                    color_continuous_scale="Viridis",)
                                return fig
                                # Generate bar chart for non-numeric data
    def generate_bar_chart(data, column):
    counts = data[column].value_counts()
    bar_chart_fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={column: "Count", "index": column},
        title=f"Counts of {column}",
    )
    return bar_chart_fig


# Generate time series plot using Plotly Express
def generate_time_series_plot(data):
    time_series_fig = px.line(
        data,
        x=data.columns[0],
        y=data.columns[1],
        labels={data.columns[0]: "Time", data.columns[1]: "Value"},
    )
    return time_series_fig


# Generate box plot using Plotly Express
def generate_box_plot(data, color_column):
    box_plot_fig = px.box(
        data,
        x=color_column,
        y=data.columns[1],
        color=color_column,
        labels={data.columns[1]: "Numerical Value"},
    )
    return box_plot_fig


def generate_linear_regression_plot(data, selected_columns):
    # Simple Linear Regression

    # Check for missing values
    if data[selected_columns].isnull().any().any():
        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        data[selected_columns] = imputer.fit_transform(data[selected_columns])

    X = data[selected_columns[0]].values.reshape(-1, 1)
    y = data[selected_columns[1]].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    # Scatterplot
    scatter_fig = generate_scatterplot(data, selected_columns[0], selected_columns[1])

    # Regression Line
    regression_line = go.Scatter(
        x=X_test.flatten(), y=y_pred.flatten(), mode="lines", name="Regression Line"
    )

    linear_regression_fig = go.Figure(data=[scatter_fig.data[0], regression_line])

    linear_regression_fig.update_layout(
        title=f"Linear Regression (MSE: {mse:.2f})",
        xaxis_title=selected_columns[0],
        yaxis_title=selected_columns[1],
    )

    return linear_regression_fig


if __name__ == "__main__":
    main()
