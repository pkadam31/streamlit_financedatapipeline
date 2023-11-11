import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType
import pandas as pd
import json
import io
import base64
from pyspark.sql.functions import col
import openai

# Initialize Spark session
spark = SparkSession.builder.master("local[*]").appName("DataTransformationApp").getOrCreate()

# Function to upload and display a file using PySpark
def upload_file():
    uploaded_file = st.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df = spark.read.option("header", "true").csv(uploaded_file)
        elif uploaded_file.type == "application/octet-stream":
            df = spark.read.parquet(uploaded_file)
        df.show()
        return df
    return None

# Function to apply transformations from JSON using PySpark
def transform_dataframe(df):
    json_file = st.file_uploader("Upload JSON for transformation", type=["json"])
    if json_file is not None:
        transformation = json.load(json_file)
        column = transformation["column"]
        final_datatype = transformation["final_datatype"]
        if final_datatype == "string":
            df = df.withColumn(column, col(column).cast(StringType()))
        elif final_datatype == "int":
            df = df.withColumn(column, col(column).cast(IntegerType()))
        df.show()
        return df
    return df

# Function to provide an option to download a dataframe as CSV
def download_csv(df):
    if st.button('Download Data as CSV'):
        pd_df = df.toPandas()
        csv = pd_df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="transformed_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

# Function to send the dataframe to OpenAI's GPT-4
def analyze_with_gpt4(df):
    # Convert DataFrame to JSON or another suitable format
    json_data = df.toJSON().collect()
    # Prepare the data to be sent in a format accepted by the API
    # Note: You may need to summarize or condense the data if it's too large

    # Sending data to GPT-4 for analysis (example, modify as needed)
    openai_api_key = st.secrets["openai_apikey"]
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        model="text-davinci-004",
        prompt=f"Analyze the following data: {json_data}",
        temperature=0.5,
        max_tokens=100
    )
    st.write(response.choices[0].text)

if __name__ == "__main__":
    st.title("Data Transformation and Analysis App with PySpark")

    # Upload and display file
    df = upload_file()

    # Apply transformations if dataframe is uploaded
    if df is not None:
        df_transformed = transform_dataframe(df)
        if df_transformed is not None:
            download_csv(df_transformed)
            analyze_with_gpt4(df_transformed)

    # Close the Spark session when done
    spark.stop()
