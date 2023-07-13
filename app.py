import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings

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
    embeddings = OpenAIEmbeddings(client='gpt3')
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with PDF', page_icon=':book:', layout='wide')
    st.header('Chat with PDF :book:')    
    st.subheader('Upload your PDF file and start chatting with it!')
    st.text_input('Enter your message here:')
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
if __name__ == '__main__':
    main()