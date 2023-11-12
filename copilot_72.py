import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import base64

# Initialize OpenAI API key
openai_api_key = st.secrets["openai_apikey"]

# Function to upload and display a file
def upload_file():
    uploaded_file = st.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/octet-stream":
            df = pd.read_parquet(uploaded_file, engine='pyarrow')
        st.write(df)
        return df
    return None

# Function to apply transformations from JSON
def transform_dataframe(df):
    json_file = st.file_uploader("Upload JSON for transformation", type=["json"])
    if json_file is not None:
        transformation = json.load(json_file)
        column = transformation["column"]
        final_datatype = transformation["final_datatype"]
        df[column] = df[column].astype(final_datatype)
        st.write(df)
        return df
    return df


def aggregate_data(df):
    st.subheader("Data Aggregation")
    agg_option = st.selectbox("Choose an aggregation operation", ["Count", "Average"])

    if agg_option == "Count":
        count = len(df)
        st.write(f"Count of rows in the DataFrame: {count}")
    elif agg_option == "Average":
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if numeric_cols.empty:
            st.write("No numeric columns available for average calculation.")
        else:
            column_to_avg = st.selectbox("Select a column to calculate the average", numeric_cols)
            if column_to_avg:
                average = df[column_to_avg].mean()
                st.write(f"Average of {column_to_avg}: {average}")

# Function to download a dataframe as CSV
def download_csv(df):
    if st.button('Download Data as CSV'):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="transformed_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

# Function to send dataframe to GPT-4 for analysis
def analyze_with_gpt4(df):
    if st.button('Analyze Data with GPT-4'):
        # Prepare a summary or key information from the dataframe
        summary = df.describe().to_json()  # Example: Sending a summary
        client = OpenAI(api_key=openai_api_key)

        # Construct the message for GPT-4
        message = {"role": "system", "content": f"Analyze this data summary and provide insights: {summary}"}

        # Send the data to GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[message]
        )

        # Display the response from GPT-4
        st.write(response.choices[0].message.content)


if __name__ == "__main__":
    st.title("Data Transformation and Analysis App")

    # Upload and display file
    df = upload_file()

    # Apply transformations if dataframe is uploaded
    if df is not None:
        df_transformed = transform_dataframe(df)
        if df_transformed is not None:
            # Download option for transformed dataframe
            aggregate_data(df_transformed)
            # Analyze with GPT-4
            analyze_with_gpt4(df_transformed)
            download_csv(df_transformed)

