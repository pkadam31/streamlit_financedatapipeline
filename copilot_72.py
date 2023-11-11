import streamlit as st
import pandas as pd
import json
import openai
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
        # Preprocess the dataframe (e.g., summarize, select key columns) before sending
        summary = df.describe().to_json()  # Example: Sending a summary
        response = openai.Completion.create(
            engine="text-davinci-004",
            prompt=f"Analyze this data: {summary}",
            max_tokens=150,
            api_key=openai_api_key
        )
        st.write(response.choices[0].text)

if __name__ == "__main__":
    st.title("Data Transformation and Analysis App")

    # Upload and display file
    df = upload_file()

    # Apply transformations if dataframe is uploaded
    if df is not None:
        df_transformed = transform_dataframe(df)
        if df_transformed is not None:
            # Download option for transformed dataframe
            download_csv(df_transformed)

            # Analyze with GPT-4
            analyze_with_gpt4(df_transformed)
