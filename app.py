import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


os.environ['OPENAI_API_KEY'] = ''


def main():
    st.header('Chat with  PDF')
    pdf = st.file_uploader("Upload your PDF File", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]
        st.write(store_name)
        
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embeddings)
        st.write('Embeddings Created')
        query = st.text_input("Ask Question from your PDF File")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
if __name__ == '__main__':
    main()