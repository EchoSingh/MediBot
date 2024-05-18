import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Initialize the LLM model
llama_model = CTransformers(
    model="C:\\Users\\adity\\Downloads\\llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type="llama",
    config={'max_new_tokens': 1000, 'temperature': 0.75, 'context_length': 2000}
)

# Load documents
loader = PyPDFDirectoryLoader('C:\\Users\\adity\\Desktop\\Chat UI\\Data')
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

# Initialize embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Initialize vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

# Initialize the ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(llama_model, vectorstore.as_retriever(), return_source_documents=True)

# Streamlit UI setup
st.set_page_config(page_title="Chatbot Interface", page_icon=":speech_balloon:", layout="wide")

st.title("Chatbot Interface")
st.markdown("""
<style>
.chat-container {
    max-width: 700px;
    margin: auto;
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
}
.user-message {
    background-color: #007bff;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.bot-message {
    background-color: #eeeeee;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

chat_history = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def md(t):
    st.markdown(t, unsafe_allow_html=True)

def display_chat():
    for msg in st.session_state['messages']:
        if msg['type'] == 'user':
            st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)

with st.form(key='chat_form'):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        st.session_state['messages'].append({"type": "user", "content": user_input})
        result = chain({"question": user_input, "chat_history": chat_history})
        st.session_state['messages'].append({"type": "bot", "content": result['answer']})
        chat_history.append({"question": user_input, "answer": result['answer']})

display_chat()

st.markdown('</div>', unsafe_allow_html=True)
