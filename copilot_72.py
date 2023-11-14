import sqlalchemy
import streamlit as st
import pandas as pd
from urllib.parse import quote_plus
from openai import OpenAI
import base64
import json

openai_api_key = st.secrets["openai_apikey"]
gcp_postgres_host = st.secrets["pg_host"]
gcp_postgres_user = st.secrets["pg_user"]
gcp_postgres_password = st.secrets["pg_password"]
gcp_postgres_dbname = st.secrets["pg_db"]


def create_table_in_postgres(df, table_name):
    """
        Uploads a pandas DataFrame to a PostgreSQL database as a new or appended table.

        :param df: DataFrame to be uploaded.
        :param table_name: Name of the table to create or append in the database.
        :return: None. Displays success or error message in Streamlit.
        """
    try:

        safe_username = quote_plus(gcp_postgres_user)
        safe_password = quote_plus(gcp_postgres_password)
        safe_host = quote_plus(gcp_postgres_host)
        safe_dbname = quote_plus(gcp_postgres_dbname)

        engine = sqlalchemy.create_engine(
            f'postgresql+psycopg2://{safe_username}:{safe_password}@{safe_host}/{safe_dbname}'
        )
        # Breaks large files into chunks for a more fault-tolerant data transfer
        chunksize = int(len(df) / 10)  # TODO: len(df) done twice
        df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=chunksize)
        st.success(f"Table '{table_name}' updated successfully in PostgreSQL")

    except Exception as e:
        st.error(f"An error occurred: {e}")


def on_checkbox_change():
    st.session_state['confirm_sensitive_data'] = st.session_state.checkbox_value


def upload_file():
    """
    Uploads a CSV or Parquet file using Streamlit's file uploader and displays its content.
    :return: The DataFrame created from the uploaded file, or None if no file is uploaded.
    """
    st.subheader("Upload your data here")

    uploaded_file = st.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension.lower() == "parquet":
            df = pd.read_parquet(uploaded_file, engine='pyarrow')
        st.write(df.head(10))
        st.write(df.dtypes)
        return df
    return None


def load_app_config():
    """
    Loads application configuration from a JSON file.
    :return: A dictionary containing configuration data.
    """
    with open('sample_transformation.json') as config_file:
        return json.load(config_file)


def transform_dataframe(df):
    """
    Transforms a DataFrame based on user-provided JSON configurations.

    The function allows users to upload a JSON file specifying various
    transformations (like fillna, astype, map, etc.) to be applied to the DataFrame.
    It also displays the original and transformed DataFrame for comparison.

    :param df: The original pandas DataFrame to be transformed.
    :return: A transformed pandas DataFrame based on the JSON configurations.
    """
    st.subheader("Data Transformation")

    # sample_transformations = load_app_config()
    # sample_transformations_str = json.dumps(sample_transformations, indent=4)
    # st.download_button("Download Sample JSON file", sample_transformations_str, "sample_transformation.json",
    #                    "text/plain")

    json_file = st.file_uploader("Upload JSON for transformation", type=["json"])
    if json_file:
        df_transformed = df.copy()
        df = df.head(10)

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

    st.write(df_transformed.dtypes)

    return df_transformed


def aggregate_data(df):
    """
    Performs basic aggregation operations (count of rows, average of a column) on a DataFrame.

    The function allows users to interactively compute the count of rows in the DataFrame
    and calculate the average of a selected numeric column. It uses Streamlit widgets for
    user interaction and displays the results.

    :param df: The pandas DataFrame on which aggregation operations will be performed.
    :return: None. The function outputs results directly to the Streamlit interface.
    """

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


def download_csv(df):
    """
    Enables downloading the provided DataFrame as a CSV file.

    The function creates a download link in the Streamlit interface, allowing users to
    download the current state of the DataFrame in CSV format.

    :param df: The pandas DataFrame to be downloaded as CSV.
    :return: None. The function creates a download link in the Streamlit interface.
    """
    if st.button('Download Data as CSV'):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="transformed_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)


def analyze_with_gpt4(df):
    """
    Sends a DataFrame's summary to GPT-4 for analysis and displays the response.

    The function summarizes the DataFrame using df.describe(include='all'), converts this summary
    to JSON, and sends it to GPT-4 for analysis. The response from GPT-4 is then displayed in
    the Streamlit interface.

    :param df: The pandas DataFrame to be analyzed by GPT-4.
    :return: None. The function outputs GPT-4's response to the Streamlit interface.
    """

    if st.button('Analyze Data with GPT-4'):
        summary = df.describe(include='all').to_json()  # Example: Sending a summary
        client = OpenAI(api_key=openai_api_key)

        # Construct the message for GPT-4
        message = {"role": "system", "content": f"Analyze this data summary and provide insights: {summary}"}

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[message]
        )

        # Display the response from GPT-4
        st.write(response.choices[0].message.content)


if __name__ == "__main__":
    st.markdown("""
            <style>
            body {
                color: #fff;  /* Adjust text color if needed */
                background-color: navy;
            }
            </style>
            """, unsafe_allow_html=True)

    st.title("Copilot72")
    st.subheader("Our very own data-savvy AI Copilot to accelerate productivity!")

    df = upload_file()

    if df is not None:
        df_transformed = transform_dataframe(df)
        if df_transformed is not None:
            aggregate_data(df_transformed)
            st.subheader("Copilot AI Data Analysis")
            analyze_with_gpt4(df_transformed)
            st.subheader("Data Export")
            download_csv(df_transformed)

            st.subheader("Update the DB - create or append your data into the Copilot72DB")

            if st.button('Create table in postgres'):
                st.warning(
                    'Please confirm that no PII, PHI, or CCI data is present in an unencrypted or unobfuscated state.')
                st.checkbox('I confirm that no sensitive data is being published in plain form',
                            value=st.session_state.get('confirm_sensitive_data', False),
                            key='checkbox_value',
                            on_change=on_checkbox_change)

            if st.session_state.get('confirm_sensitive_data', False):
                table_name = st.text_input("Enter the name of the table to create in PostgreSQL:")
                # Button to publish the table
                if st.button('Publish table'):
                    if table_name:  # Check if table_name is provided
                        create_table_in_postgres(df, table_name)
                    else:
                        st.error("Please enter a table name.")
