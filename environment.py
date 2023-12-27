import streamlit as st
import pandas as pd
import plotly.express as px

def load_data(file_uploader):
    try:
        df = pd.read_csv(file_uploader)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the data: {str(e)}")
        return None

def visualize_pie_chart(df):
    selected_column = st.selectbox("Select a column for the pie chart", df.columns)
    
    if selected_column:
        try:
            pie_data = df[selected_column].value_counts().reset_index()
            pie_data.columns = ['Category', 'Count']

            fig = px.pie(pie_data, names='Category', values='Count', title=f'Pie Chart - {selected_column}',
                         hover_data=['Count'], labels={'Count': 'Frequency'})

            # Customize layout
            fig.update_traces(textinfo='percent+label', pull=[0.1] * len(pie_data), hole=0.3)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"An error occurred while visualizing the pie chart: {str(e)}")
    else:
        st.warning("Please select a column.")

def visualize_bar_chart(df):
    try:
        selected_columns = st.multiselect("Select columns for Y-axis", df.columns)
        if selected_columns:
            st.bar_chart(df[selected_columns])
        else:
            st.warning("Please select at least one column.")
    except Exception as e:
        st.error(f"An error occurred while visualizing the bar chart: {str(e)}")

def visualize_scatter_plot(df):
    try:
        x_column = st.selectbox("Select X-axis column", df.columns)
        y_column = st.selectbox("Select Y-axis column", df.columns)
        color_column = st.selectbox("Select color column (optional)", df.columns, key='color_column')

        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title="Scatter Plot",
                         labels={x_column: "X-axis", y_column: "Y-axis"},
                         hover_name=df.index,
                         template="plotly_white")

        fig.update_traces(marker_size=8)
        st.plotly_chart(fig)

        # Show descriptive statistics
        st.write("Descriptive Statistics:")
        st.write(df.describe())
    except Exception as e:
        st.error(f"An error occurred while visualizing the scatter plot: {str(e)}")

def main():
    st.title("SoftGrow Data Visualization")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            chart_type = st.selectbox("Select a chart type", [
                "Bar Chart",
                "Pie Chart",
                "Scatter Plot",
            ])

            if chart_type == "Bar Chart":
                visualize_bar_chart(df)
            elif chart_type == "Pie Chart":
                visualize_pie_chart(df)
            elif chart_type == "Scatter Plot":
                visualize_scatter_plot(df)

            st.write("Data Table:")
            st.dataframe(df)

if __name__ == "__main__":
    main()
