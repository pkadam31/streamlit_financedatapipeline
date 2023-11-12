import streamlit as st
import pandas as pd
import psycopg2
import pyarrow
from openai import OpenAI
import base64
import json

openai_api_key = st.secrets["openai_apikey"]
gcp_postgres_host = st.secrets["pg_host"]
gcp_postgres_user = st.secrets["pg_user"]
gcp_postgres_password = st.secrets["pg_password"]
gcp_postgres_dbname = st.secrets["pg_db"]


def get_db_connection():
    """
    Establishes a connection to the database using global connection parameters.
    :return: The database connection object.
    """
    return psycopg2.connect(
        host=gcp_postgres_host,
        user=gcp_postgres_user,
        password=gcp_postgres_password,
        dbname=gcp_postgres_dbname
    )


def execute_sql_query(cursor, sql_query):
    """
    Executes the provided SQL query and returns the results.
    :param cursor: The database cursor object.
    :param sql_query: The SQL query to execute.
    :return: A tuple containing the raw result and a DataFrame representation of the result.
    """
    cursor.execute(sql_query)
    result = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    return result, pd.DataFrame(result, columns=column_names)


def close_db_connection(conn, cursor=None):
    """
    Closes the database connection and cursor if provided.
    :param conn: The database connection object.
    :param cursor: The database cursor object. Default is None.
    """
    if cursor:
        cursor.close()
    if conn:
        conn.close()

# Function to upload and display a file
def upload_file():
    st.subheader("Upload your data here")

    uploaded_file = st.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/octet-stream":
            df = pd.read_parquet(uploaded_file, engine='pyarrow')
        st.write(df.head(25))
        st.write(df.dtypes)
        return df
    return None


def transform_dataframe(df):
    st.subheader("Data Transformation")

    json_file = st.file_uploader("Upload JSON for transformation", type=["json"])
    if json_file is not None:
        transformations = json.load(json_file)
        for column, transform_list in transformations.items():
            for transform in transform_list:
                if 'map' in transform:
                    # Check if the column's data type is numeric
                    if pd.api.types.is_numeric_dtype(df[column]):
                        # Convert string keys in the map to the column's numeric type
                        map_transformation = {df[column].dtype.type(k): v for k, v in transform['map'].items()}
                    else:
                        map_transformation = transform['map']

                    original_values = df[column].copy()
                    df[column] = df[column].map(map_transformation).fillna(original_values)

                elif 'astype' in transform:
                    df[column] = df[column].astype(transform['astype'])
        st.write(df.head(25))
        return df
    return df


def aggregate_data(df):
    st.subheader("Data Aggregation")

    # Count operation
    if st.button("Compute Count of Rows"):
        count = len(df)
        st.write(f"Count of rows in the DataFrame: {count}")

    # Average operation
    st.write("Calculate Average of a Column")
    numeric_cols = df.select_dtypes(exclude=['object', 'bool', 'datetime64[ns]']).columns
    if numeric_cols.empty:
        st.write("No numeric columns available for average calculation.")
    else:
        column_to_avg = st.selectbox("Select a column to calculate the average", numeric_cols)
        if st.button("Compute Average"):
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
    st.markdown("""
        <style>
        .title {
            color: navy;
        }
        .subheader {
            color: navy;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Copilot72")
    st.subheader("Our very own data-savvy AI Copilot to accelerate productivity!")

    # # Upload and display file
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

