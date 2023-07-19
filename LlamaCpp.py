import os
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp, HuggingFaceHub
from langchain import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer: {context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template,)

load_dotenv()
persist_directory = os.environ.get('PERSIST_DIRECTORY')
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',1024))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriver = db.as_retriever(search_kwargs={"k": target_source_chunks})
    return retriver 

def get_conversation_chain(retriever,model_path, QA_CHAIN_PROMPT):
    llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch,verbose=True)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        memory=memory,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        retriever=retriever,
        
    )
    return chain

def handle_userinput(user_question):

    if st.session_state.conversation is None:
        st.warning("Please load the Vectorstore first!")
        return
    else:
        with st.spinner('Thinking...',):
            response = st.session_state.conversation({'query': user_question})
        
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    
    st.set_page_config(page_title='Chat with PDF', page_icon=':book:', layout='wide', )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header('Chat with PDF :robot_face:')    
    st.subheader('Upload your PDF file and start chatting with it!')
    user_question = st.text_input('Enter your message here:')
    if st.button('Load Vectorstore :desktop_computer:'):
        with st.spinner('Loading Vectorstore...'):
            vector_store = get_vector_store()
            st.session_state.conversation = get_conversation_chain(retriever=vector_store, model_path=model_path, QA_CHAIN_PROMPT=QA_CHAIN_PROMPT)
    
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
