import sqlalchemy
import streamlit as st
import pandas as pd
import psycopg2
import pyarrow
from urllib.parse import quote_plus
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


def create_table_in_postgres(df, table_name):
    try:

        safe_username = quote_plus(gcp_postgres_user)
        safe_password = quote_plus(gcp_postgres_password)

        engine = sqlalchemy.create_engine(
            f'postgresql+psycopg2://{safe_username}:{safe_password}@{gcp_postgres_host}/{gcp_postgres_dbname}'
        )

        df.to_sql(table_name, engine, if_exists='append', index=False)
        st.success(f"Table '{table_name}' updated successfully in PostgreSQL")
    except Exception as e:
        st.error(f"An error occurred: {e}")


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
        st.write(df.head(10))
        st.write(df.dtypes)
        return df
    return None


def transform_dataframe(df):
    st.subheader("Data Transformation")

    df_transformed = df.copy()
    df = df.head(10)

    json_file = st.file_uploader("Upload JSON for transformation", type=["json"])
    if json_file:
        transformations = json.load(json_file)

        if 'fillna' in transformations:
            for column, value in transformations['fillna'].items():
                df_transformed[column] = df_transformed[column].fillna(value)

        if 'astype' in transformations:
            for column, dtype in transformations['astype'].items():
                df_transformed[column] = df_transformed[column].astype(dtype)

        if 'map' in transformations:
            for column, mapping in transformations['map'].items():
                original_values = df_transformed[column].copy()
                if pd.api.types.is_numeric_dtype(original_values):
                    map_transformation = {original_values.dtype.type(k): v for k, v in mapping.items()}
                else:
                    map_transformation = mapping

                df_transformed[column] = original_values.map(map_transformation).fillna(original_values)

        if transformations.get('drop_duplicates'):
            df_transformed = df_transformed.drop_duplicates()

        if 'sort_values' in transformations:
            sort_params = transformations['sort_values']
            df_transformed = df_transformed.sort_values(by=sort_params['by'], ascending=sort_params['ascending'])

        if 'rename' in transformations:
            df_transformed = df_transformed.rename(columns=transformations['rename'])

    # Display the original top 10 rows for comparison
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original DataFrame (Top 10 Rows):")
        st.dataframe(df)

    with col2:
        st.write("Transformed DataFrame:")
        st.dataframe(df_transformed.head(10))

    return df_transformed


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

            if st.button('Create table in postgres'):
                create_table_in_postgres(df, 'test')

