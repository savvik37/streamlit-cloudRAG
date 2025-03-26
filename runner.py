import streamlit as st
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent
#import PyPDF2
import pandas as pd
#import base64
from eml_parser import EmlParser  # Import your client module here, assuming it's already defined

# Define the vector store ID and system message
client = OpenAI(
    api_key = st.secrets["openai"]["api_key"]
)
vector_store_id = st.secrets["vector_store"]["api_key"]
sysmes = "SYSTEM MESSAGE: You are a helpful assistant that uses a vector store to provide accurate and relevant answers to user queries. Always refer to the vector store for context. Your vector store contains emails sent by Sava (or Savvik, or Sav are nicknames) and you answer questions based on what the emails contain."

# Streamlit app layout
st.title('Sava Email OpenAI Vector Store RAG')
st.write('Enter your query below.')

# Input text box for the user query
user_query = st.text_input('Query', '')

# Button to send the query
if st.button('Get Response'):
    if user_query:
        # Construct the query by combining the system message and user query
        query = sysmes + " " + user_query
        
        # Get the response from the vector store
        response = client.responses.create(
            input=query,
            model="gpt-4o",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }]
        )

        # Display the response from the vector store
        st.write("Response: ")
        st.write(str(response.output[len(response.output) - 1].content[0].text))  # Assuming 'response' contains a 'data' field with the result
    else:
        st.warning("Please enter a query before submitting.")

