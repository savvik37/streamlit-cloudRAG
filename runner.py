import streamlit as st
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key=st.secrets["openai"]["api_key"]
)
vector_store_id = st.secrets["vector_store"]["id"]
sysmes = "SYSTEM MESSAGE: You are a helpful assistant that uses a vector store to provide accurate and relevant answers to user queries. Always refer to the vector store for context. Your vector store contains emails sent by Sava (or Savvik, or Sav are nicknames) and you answer questions based on what the emails contain."

# Page configuration
st.set_page_config(
    page_title="Sava's Email Assistant",
    page_icon="📧",
    layout="wide"
)

# Sidebar with information
with st.sidebar:
    st.title("📧 Sava's Email Assistant")
    st.markdown("""
    This app lets you query Sava's email archive using natural language.
    
    **How it works:**
    1. Type your question in the chat input
    2. The system searches through the email vector store
    3. It provides an answer based on relevant emails
    
    Powered by OpenAI's vector search technology.
    """)
    
    # Add a reset button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.experimental_rerun()

# Main content area
st.title("Sava's Email Assistant")

# Initialize chat message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Sava's email assistant. How can I help you today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_query = st.chat_input("Ask about Sava's emails...")

# Process user input
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Prepare the full context from chat history
    chat_context = ""
    for message in st.session_state.messages:
        prefix = "User: " if message["role"] == "user" else "Assistant: "
        chat_context += f"{prefix}{message['content']}\n"
    
    # Combine system message with chat context
    full_query = f"{sysmes}\n\n{chat_context}"
    
    # Get response from OpenAI with vector store retrieval
    with st.chat_message("assistant"):
        with st.spinner("Searching emails..."):
            try:
                response = client.responses.create(
                    input=full_query,
                    model="gpt-4o",
                    tools=[{
                        "type": "file_search",
                        "vector_store_ids": [vector_store_id],
                    }]
                )
                
                # Extract the answer
                answer = str(response.output[len(response.output) - 1].content[0].text)
                
                # Display the answer
                st.write(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})