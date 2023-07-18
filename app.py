import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv('.env')

@st.cache(allow_output_mutation=True)
def load_documents():
    documents = []
    # Create a List of Documents from all of our files in the ./docs folder
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            pdf_path = os.path.join("docs", file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = os.path.join("docs", file)
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = os.path.join("docs", file)
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)

    # Convert the document chunks to embeddings and save them to the vector store
    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
    vectordb.persist()

    return vectordb

def main():
    st.set_page_config(page_title="Chat with Multiple Documents", page_icon=":books:")
    st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)

    st.header("Chat with Multiple Documents :books:")

    # Load or create the vector store
    vectorstore = load_documents()

    # Create the conversation chain
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k'),
        retriever=vectorstore.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your documents:")

    if st.button("Ask"):
        if user_question:
            response = pdf_qa(
                {"question": user_question, "chat_history": st.session_state.chat_history}
            )
            st.session_state.chat_history.append(
                (user_question, response["answer"])
            )

    for query, answer in st.session_state.chat_history:
        if query:
            st.write(f"User: {query}")
        st.write(f"Bot: {answer}")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs, DOCs, or TXTs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # Extract text from uploaded documents
                    raw_text = ""
                    for pdf in pdf_docs:
                        if pdf.name.endswith(".pdf"):
                            pdf_reader = PyPDFLoader(pdf)
                            raw_text += " ".join(pdf_reader.load())
                        elif pdf.name.endswith('.docx') or pdf.name.endswith('.doc'):
                            doc_reader = Docx2txtLoader(pdf)
                            raw_text += " ".join(doc_reader.load())
                        elif pdf.name.endswith('.txt'):
                            text_reader = TextLoader(pdf)
                            raw_text += " ".join(text_reader.load())

                    # Split the text into chunks
                    text_chunks = CharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=10).split_text(raw_text)

                    # Update the vector store
                    vectorstore.update_from_texts(texts=text_chunks)

                    # Update the conversation chain
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 6}),
                        return_source_documents=True,
                        verbose=False
                    )

if __name__ == '__main__':
    main()
