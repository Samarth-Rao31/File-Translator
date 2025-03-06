import streamlit as st
import time

from PyPDF2 import PdfReader
from googletrans import Translator
from languages import *

def translate_japanese_text(pdfextraction_text, dest_language):
    translator = Translator()
    translation = translator.translate(pdfextraction_text, dest=dest_language)
    return translation.text

# Extract text from the selected page
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        print('Aarna1', pdf)
        time.sleep(15)
        pdf_reader= PdfReader(pdf)
        print('Aarna2', pdf_reader)
        time.sleep(30)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def main():
    st.header("Y-ChatGPT", divider='rainbow')

    # Upload PDF file through Streamlit

    # user_question = st.text_input("Ask a Question from the PDF Files")

    # if user_question:
        # user_input(user_question)

        # pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

        # st.subheader("Select a page to translate:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True)

    # if st.button("Submit"):
    #     with st.spinner("Processing..."):
    #         time.sleep(30)
    #         raw_text = get_pdf_text(pdf_docs)
    #         st.success("Done")

    # if pdf_docs is not None:

        # selected_page = st.selectbox("Page", range(pdf_document.page_count), key="page_selector")

        # page = pdf_document[selected_page]
        # japanese_text = page.get_text()

        # with st.sidebar:
        #     st.title("Menu:")

        # # Display the original text
        # st.subheader("Original Japanese Text:")
        # st.text_area("Japanese Text", value=japanese_text, height=200)

        # Allow the user to choose the destination language
    target_language = st.selectbox("Select target language:", languages)

    if st.button("Submit"):
        with st.spinner("Processing..."):
            # Translate the Japanese text to the selected language
            pdfextraction_text = get_pdf_text(pdf_docs)

            # Try to convert into txt file and place and test it
            print('Samarth', pdfextraction_text)
            time.sleep(30)
            translated_text = translate_japanese_text(pdfextraction_text, target_language)

            # Display the translated text
            st.subheader(f"Translated Text ({target_language}):")
            st.text_area("Translated Text", value=translated_text, height=200)

if __name__ == "__main__":
    main()
