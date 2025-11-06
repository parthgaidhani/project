import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from streamlit import session_state
import numpy as np

# Initialize session state
def init_session_state():
    if "user_email" not in session_state:
        session_state.user_email = None

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
            scatterplot_data = process_data_file(data_file)

            if scatterplot_data is None or scatterplot_data.empty:
                st.error("No data available to display.")
                return

            # Data Insights Section
            st.subheader("Data Insights")
            display_data_insights(scatterplot_data)

            # Data Exploration
            st.subheader("Data Exploration")
            st.write("**Dataset Preview:**")
            st.dataframe(scatterplot_data.head(10))
            
            st.write("**Statistical Summary:**")
            st.write(scatterplot_data.describe())

            # Get numeric and categorical columns
            numeric_cols = scatterplot_data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = scatterplot_data.select_dtypes(include=['object', 'category']).columns.tolist()

            # Scatterplot (only if we have at least 2 numeric columns)
            if len(numeric_cols) >= 2:
                st.subheader("Scatterplot")
                scatterplot_fig = generate_scatterplot(scatterplot_data, numeric_cols)
                if scatterplot_fig:
                    st.plotly_chart(scatterplot_fig, key="scatterplot", use_container_width=True)

            # Correlation Heatmap (only for numeric data)
            if len(numeric_cols) >= 2:
                st.subheader("Correlation Heatmap")
                heatmap_fig = generate_heatmap(scatterplot_data, numeric_cols)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, key="heatmap", use_container_width=True)

            # Bar Charts for Categorical Data
            if categorical_cols:
                st.subheader("Categorical Data Distribution")
                for column in categorical_cols[:5]:  # Limit to first 5 categorical columns
                    bar_chart_fig = generate_bar_chart(scatterplot_data, column)
                    if bar_chart_fig:
                        st.plotly_chart(bar_chart_fig, key=f"barchart_{column}", use_container_width=True)

            # Pair Plot (only if we have 2-5 numeric columns)
            if 2 <= len(numeric_cols) <= 5:
                st.subheader("Pair Plot")
                pair_plot_fig = generate_pair_plot(scatterplot_data, numeric_cols)
                if pair_plot_fig:
                    st.plotly_chart(pair_plot_fig, key="pair_plot", use_container_width=True)

            # Histograms for Numeric Columns
            if numeric_cols:
                st.subheader("Distribution Histograms")
                for column in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    histogram_fig = generate_histogram(scatterplot_data, column)
                    if histogram_fig:
                        st.plotly_chart(histogram_fig, key=f"histogram_{column}", use_container_width=True)

            # 3D Scatter Plot
            if len(numeric_cols) >= 3:
                st.subheader("3D Scatter Plot")
                scatterplot_3d_fig = generate_3d_scatter(scatterplot_data, numeric_cols)
                if scatterplot_3d_fig:
                    st.plotly_chart(scatterplot_3d_fig, key="scatterplot_3d", use_container_width=True)

            # Time Series Plot
            date_col = detect_date_column(scatterplot_data)
            if date_col and len(numeric_cols) > 0:
                st.subheader("Time Series Plot")
                time_series_fig = generate_time_series_plot(scatterplot_data, date_col, numeric_cols[0])
                if time_series_fig:
                    st.plotly_chart(time_series_fig, key="time_series_plot", use_container_width=True)

            # Box Plot
            if len(numeric_cols) >= 1 and (len(categorical_cols) >= 1 or len(numeric_cols) >= 2):
                st.subheader("Box Plot")
                box_plot_fig = generate_box_plot(scatterplot_data, numeric_cols, categorical_cols)
                if box_plot_fig:
                    st.plotly_chart(box_plot_fig, key="box_plot", use_container_width=True)

            # Linear Regression
            if len(numeric_cols) >= 2:
                st.subheader("Linear Regression Analysis")
                linear_regression_fig, metrics = generate_linear_regression_plot(scatterplot_data, numeric_cols)
                if linear_regression_fig:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MSE", f"{metrics['mse']:.4f}")
                    with col2:
                        st.metric("R² Score", f"{metrics['r2']:.4f}")
                    with col3:
                        st.metric("Slope", f"{metrics['slope']:.4f}")
                    st.plotly_chart(linear_regression_fig, key="linear_regression", use_container_width=True)

            # Interactive Controls
            if numeric_cols:
                st.sidebar.subheader("Interactive Controls")
                selected_x = st.sidebar.selectbox("Select X-axis", numeric_cols, key="x_axis")
                remaining_cols = [col for col in numeric_cols if col != selected_x]
                if remaining_cols:
                    selected_y = st.sidebar.selectbox("Select Y-axis", remaining_cols, key="y_axis")
                    
                    color_col = None
                    size_col = None
                    
                    if categorical_cols:
                        use_color = st.sidebar.checkbox("Color by category")
                        if use_color:
                            color_col = st.sidebar.selectbox("Select color column", categorical_cols)
                    
                    if len(numeric_cols) > 2:
                        use_size = st.sidebar.checkbox("Size by numeric value")
                        if use_size:
                            size_options = [col for col in numeric_cols if col not in [selected_x, selected_y]]
                            if size_options:
                                size_col = st.sidebar.selectbox("Select size column", size_options)
                    
                    st.subheader("Custom Scatterplot")
                    custom_scatterplot_fig = generate_custom_scatterplot(
                        scatterplot_data, selected_x, selected_y, color_col, size_col
                    )
                    if custom_scatterplot_fig:
                        st.plotly_chart(custom_scatterplot_fig, key="custom_scatterplot", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

# Display comprehensive data insights
def display_data_insights(data):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", data.shape[0])
        st.metric("Total Columns", data.shape[1])
    
    with col2:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        st.metric("Numeric Columns", len(numeric_cols))
        st.metric("Categorical Columns", len(categorical_cols))
    
    with col3:
        missing_values = data.isnull().sum().sum()
        missing_percentage = (missing_values / (data.shape[0] * data.shape[1])) * 100
        st.metric("Missing Values", missing_values)
        st.metric("Missing %", f"{missing_percentage:.2f}%")
    
    # Column-wise missing values
    if missing_values > 0:
        st.write("**Missing Values by Column:**")
        missing_data = pd.DataFrame({
            'Column': data.columns,
            'Missing Count': data.isnull().sum(),
            'Missing %': (data.isnull().sum() / len(data) * 100).round(2)
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        if not missing_data.empty:
            st.dataframe(missing_data)
    
    # Data types
    st.write("**Data Types:**")
    dtype_df = pd.DataFrame({
        'Column': data.columns,
        'Data Type': data.dtypes.astype(str),
        'Unique Values': [data[col].nunique() for col in data.columns]
    })
    st.dataframe(dtype_df)

# Process data file and return DataFrame
def process_data_file(data_file):
    try:
        if data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(data_file)
        elif data_file.type == "text/csv":
            df = pd.read_csv(data_file)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
        
        # Basic data type optimization
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        return df
    except Exception as e:
        st.error(f"Error reading data file: {e}")
        return None

# Detect date column
def detect_date_column(data):
    date_cols = data.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    if len(date_cols) > 0:
        return date_cols[0]
    return None

# Generate scatterplot using Plotly
def generate_scatterplot(data, numeric_cols):
    try:
        if len(numeric_cols) < 2:
            return None
        fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], 
                        title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
        return fig
    except Exception as e:
        st.warning(f"Could not generate scatterplot: {e}")
        return None

# Generate custom scatterplot
def generate_custom_scatterplot(data, x_col, y_col, color_col=None, size_col=None):
    try:
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col, size=size_col,
                        title=f"{y_col} vs {x_col}")
        return fig
    except Exception as e:
        st.warning(f"Could not generate custom scatterplot: {e}")
        return None

# Generate heatmap for numeric data
def generate_heatmap(data, numeric_cols):
    try:
        if len(numeric_cols) < 2:
            return None
        correlation_matrix = data[numeric_cols].corr()
        fig = px.imshow(correlation_matrix, 
                       color_continuous_scale="RdBu_r",
                       title="Correlation Heatmap",
                       text_auto='.2f',
                       aspect="auto")
        return fig
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")
        return None

# Generate bar chart for non-numeric data
def generate_bar_chart(data, column):
    try:
        counts = data[column].value_counts().head(20)  # Limit to top 20 categories
        fig = px.bar(x=counts.index, y=counts.values, 
                    labels={'x': column, 'y': 'Count'}, 
                    title=f"Distribution of {column}")
        fig.update_xaxes(tickangle=-45)
        return fig
    except Exception as e:
        st.warning(f"Could not generate bar chart for {column}: {e}")
        return None

# Generate pair plot
def generate_pair_plot(data, numeric_cols):
    try:
        fig = px.scatter_matrix(data[numeric_cols], 
                               title="Pair Plot of Numeric Variables")
        fig.update_traces(diagonal_visible=False)
        return fig
    except Exception as e:
        st.warning(f"Could not generate pair plot: {e}")
        return None

# Generate histogram
def generate_histogram(data, column):
    try:
        fig = px.histogram(data, x=column, 
                          title=f"Distribution of {column}",
                          nbins=30)
        return fig
    except Exception as e:
        st.warning(f"Could not generate histogram for {column}: {e}")
        return None

# Generate 3D scatter plot
def generate_3d_scatter(data, numeric_cols):
    try:
        if len(numeric_cols) < 3:
            return None
        fig = px.scatter_3d(data,
                           x=numeric_cols[0],
                           y=numeric_cols[1],
                           z=numeric_cols[2],
                           title=f"3D Scatter: {numeric_cols[0]}, {numeric_cols[1]}, {numeric_cols[2]}")
        return fig
    except Exception as e:
        st.warning(f"Could not generate 3D scatter plot: {e}")
        return None

# Generate time series plot using Plotly
def generate_time_series_plot(data, date_col, value_col):
    try:
        df_sorted = data.sort_values(date_col)
        fig = px.line(df_sorted, x=date_col, y=value_col,
                     title=f"{value_col} over Time",
                     labels={date_col: "Date", value_col: "Value"})
        return fig
    except Exception as e:
        st.warning(f"Could not generate time series plot: {e}")
        return None

# Generate box plot
def generate_box_plot(data, numeric_cols, categorical_cols):
    try:
        y_col = numeric_cols[0]
        x_col = categorical_cols[0] if categorical_cols else (numeric_cols[1] if len(numeric_cols) >= 2 else None)
        
        if x_col is None:
            return None
            
        fig = px.box(data, x=x_col, y=y_col, 
                    title=f"Box Plot: {y_col} by {x_col}")
        return fig
    except Exception as e:
        st.warning(f"Could not generate box plot: {e}")
        return None

# Generate linear regression plot
def generate_linear_regression_plot(data, numeric_cols):
    try:
        if len(numeric_cols) < 2:
            return None, {}
        
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        # Prepare data
        df_clean = data[[x_col, y_col]].dropna()
        
        if len(df_clean) < 2:
            st.warning("Not enough data points for regression analysis")
            return None, {}
        
        X = df_clean[x_col].values.reshape(-1, 1)
        y = df_clean[y_col].values.reshape(-1, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        slope = model.coef_[0][0]
        
        # Create visualization
        fig = go.Figure()
        
        # Add scatter plot of all data
        fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), 
                                mode='markers', name='Data Points',
                                marker=dict(color='blue', size=8, opacity=0.6)))
        
        # Add regression line
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_range = model.predict(X_range)
        fig.add_trace(go.Scatter(x=X_range.flatten(), y=y_range.flatten(),
                                mode='lines', name='Regression Line',
                                line=dict(color='red', width=3)))
        
        fig.update_layout(title=f"Linear Regression: {y_col} vs {x_col}",
                         xaxis_title=x_col,
                         yaxis_title=y_col)
        
        metrics = {'mse': mse, 'r2': r2, 'slope': slope}
        return fig, metrics
    except Exception as e:
        st.warning(f"Could not generate linear regression plot: {e}")
        return None, {}

if __name__ == "__main__":
    main()
