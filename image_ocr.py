import os
import streamlit as st

from filetype import guess
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
# from langchain.llms import OpenAI
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

embeddings = AzureOpenAIEmbeddings(
    deployment="yokogawaconnectsharepoint",
    model="text-embedding-ada-002",
    openai_api_base="https://openai-yokogawa-internal.openai.azure.com/openai/deployments/yokogawaconnectsharepoint",
    openai_api_type="azure",
    api_key="245b9782f4744cc681465e5459c3ee18",
    # api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2023-07-01-preview"
)

print('Samarth', embeddings)

def detect_document_type(document_path):
    
    guess_file = guess(document_path)
    file_type = ""
    image_types = ['jpg', 'jpeg', 'png', 'gif']
    
    if(guess_file.extension.lower() == "pdf"):
        file_type = "pdf"
        
    elif(guess_file.extension.lower() in image_types):
        file_type = "image"
        
    else:
        file_type = "unkown"
        
    return file_type

# research_paper_path = "./data/transformer_paper.pdf"
# article_information_path = "./data/zoumana_article_information.png"

def extract_file_content(file_path):
    
    file_type = detect_document_type(file_path)
    
    if(file_type == "pdf"):
        loader = UnstructuredFileLoader(file_path)
        
    elif(file_type == "image"):
        loader = UnstructuredImageLoader(file_path)
        
    documents = loader.load()
    documents_content = '\n'.join(doc.page_content for doc in documents)
    
    return documents_content


# research_paper_content = extract_file_content(research_paper_path)
# article_information_content = extract_file_content(article_information_path)

# nb_characters = 400

# print(f"First {nb_characters} Characters of the Paper: \n{research_paper_content[:nb_characters]}...")
# print("*"*25)
# print(f"First {nb_characters} Characters of Article Information Document :\n {article_information_content[:nb_characters]}...")

def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(        
        separator = "\n\n",
        chunk_size = 1000,
        chunk_overlap  = 150,
        length_function = len,
    )

    research_paper_chunks = text_splitter.split_text(text)
    print('Mistybaby', research_paper_chunks)
    return research_paper_chunks


def get_doc_search(text_splitter):
    
    return FAISS.from_texts(text_splitter, embeddings)

chain = load_qa_chain(AzureChatOpenAI(model="gpt-35-turbo-16k",
                    deployment_name="yokogawaconnectgpt16k",
                    openai_api_base="https://openai-yokogawa-internal.openai.azure.com/openai/deployments/yokogawaconnectgpt16k",
                    openai_api_type="azure",
                    api_key="245b9782f4744cc681465e5459c3ee18",
                    # api_key=os.getenv("OPENAI_API_KEY"),
                    api_version="2023-07-01-preview",
                    temperature=0.7,
                    streaming=True), chain_type = "map_rerank",  
                    return_intermediate_steps=True)

def chat_with_file():
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    llm = AzureChatOpenAI(model="gpt-35-turbo-16k",
                    deployment_name="yokogawaconnectgpt16k",
                    openai_api_base="https://openai-yokogawa-internal.openai.azure.com/openai/deployments/yokogawaconnectgpt16k",
                    openai_api_type="azure",
                    api_key="245b9782f4744cc681465e5459c3ee18",
                    # api_key=os.getenv("OPENAI_API_KEY"),
                    api_version="2023-07-01-preview",
                    temperature=0.7,
                    streaming=True)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm, chain_type="map_rerank", prompt=prompt)

    return chain

def user_input(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # text_chunks = get_text_chunks(raw_text)
    # file_text_path = extract_file_content(file_path)
    # new_db = FAISS.load_local("faiss_index", folder_path=file_text_path, embeddings=embeddings)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = chat_with_file()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Y-ChatGPT with iMAGE💁")

    user_question = st.text_input("Ask a Question from the iMAGE")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = extract_file_content(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_doc_search(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()