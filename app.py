import langchain
langchain.verbose = False
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ''
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name='hkunlp/instructor-xl')
    vectorstore = Chroma.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(repo_id='google/flan-t5-xxl')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with PDF', page_icon=':book:', layout='wide')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header('Chat with PDF :book:')    
    st.subheader('Upload your PDF file and start chatting with it!')
    user_question = st.text_input('Enter your message here:')

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader('Your PDF documents')
        pdf_docs = st.file_uploader('Upload your PDF file here', type=['pdf'], accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing your PDF file...'):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks) 
                # create a vector store
                vector_store = get_vector_store(text_chunks)
                # crate conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
    __name__: str
if __name__ == '__main__':
    main()