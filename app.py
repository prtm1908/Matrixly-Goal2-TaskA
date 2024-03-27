import streamlit as st
from unstructured.partition.pdf import partition_pdf
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import urllib.request
from PIL import Image
import requests
import google.generativeai as genai
import os
from IPython.display import display, HTML, Markdown
import textwrap
import uuid
import io
from io import BytesIO
import re
import base64
import pickle
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
os.environ['GOOGLE_API_KEY']=GOOGLE_API_KEY


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

@st.cache_resource
def loadTextTables(file_path):
    image_path = "./"
    pdf_elements = partition_pdf(
        file_path,
        chunking_strategy="by_title",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        max_characters=3000,
        new_after_n_chars=2800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_path
        )


    def categorize_elements(raw_pdf_elements):
        text_elements = []
        table_elements = []
        for element in raw_pdf_elements:
            if 'CompositeElement' in str(type(element)):
                text_elements.append(str(element))
            elif 'Table' in str(type(element)):
                table_elements.append(str(element))
        return text_elements, table_elements

    texts, tables = categorize_elements(pdf_elements)
    print(len(texts))
    print(len(tables))


    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0, max_tokens=1024)
    model_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision",temperature=0, max_tokens=1024)

    def generate_text_summaries(texts, tables, summarize_texts=False):
        """
        Summarize text elements
        texts: List of str
        tables: List of str
        summarize_texts: Bool to summarize texts
        """

        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well-optimized for retrieval. Table \
        or text: {element} """
        prompt = PromptTemplate.from_template(prompt_text)

        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        text_summaries = []
        table_summaries = []

        if texts and summarize_texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
        elif texts:
            text_summaries = texts

        if tables:
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

        return text_summaries, table_summaries

    text_summaries, table_summaries = generate_text_summaries(texts, tables)

    return text_summaries, table_summaries, texts, tables, model, model_vision


@st.cache
def loadImages():
    def encode_image(image_path):
        """Getting the base64 string"""
        
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    def list_images(folder_path="figures"):
        images = []
        img_base64_list = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                images.append(os.path.join(folder_path, filename))
                base64_image = encode_image(os.path.join(folder_path, filename))
                img_base64_list.append(base64_image)
        return images, img_base64_list

    images, img_base64_list = list_images()

    return images, img_base64_list


def save_uploaded_file(uploaded_file, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever




def base64_to_image(base64_string, filename):
    base64_string = base64_string.split(',')[-1]
    image_bytes = base64.b64decode(base64_string)
    img = Image.open(BytesIO(image_bytes))
    img.save(filename)

def looks_like_base64(sb):
        """Check if the string looks like base64"""
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    resized_img = img.resize(size, Image.LANCZOS)

    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}
    
def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    text_message = {
        "type": "text",
        "text": (
            "You are an AI scientist tasking with providing factual answers from research papers.\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide answers related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever, model_vision):
    """
    Multi-modal RAG chain
    """

    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model_vision  # MM_LLM
        | StrOutputParser()
    )

    return chain

def main():
    # Set Streamlit title and description
    st.title("PDF Text and Table Summarizer")
    st.write("This app summarizes text and tables from a PDF.")

    # File uploader for PDF
    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button('Load') and uploaded_file:
        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file, "uploads")
        with open('file_path.pkl', 'wb') as f:
            pickle.dump(file_path, f)

    if st.button('Submit'):
        # Load file path if available
        with open('file_path.pkl', 'rb') as f:
            file_path = pickle.load(f)

        # Load text and tables from uploaded PDF using the imported function
        text_summaries, table_summaries, texts, tables, model, model_vision = loadTextTables(file_path)

        images, img_base64_list = loadImages()
        for image_path in images:
            st.image(image_path, caption='Image')

        with open('img_base64_list.pkl', 'wb') as f:
            pickle.dump(img_base64_list, f)

        image_summaries = input("Enter captions for all images (separated by commas):").split(',')
        if(image_summaries):

            with open('image_summaries.pkl', 'wb') as f:
                pickle.dump(image_summaries, f)

            with open('image_summaries.pkl', 'rb') as f:
                image_summaries = pickle.load(f)

            print(image_summaries)
            # Ask user for a query
            query = input("Enter your query: ")
            with open('query.pkl', 'wb') as f:
                pickle.dump(query, f)
    
    if st.button("Search"):
        # Load file path and query if available
        with open('file_path.pkl', 'rb') as f:
            file_path = pickle.load(f)
        
        with open('query.pkl', 'rb') as f:
            query = pickle.load(f)

        with open('img_base64_list.pkl', 'rb') as f:
            img_base64_list = pickle.load(f)

        # Load text and tables from uploaded PDF using the imported function
        text_summaries, table_summaries, texts, tables, model, model_vision = loadTextTables(file_path)

        # Check if image summaries are available
        if os.path.exists('image_summaries.pkl'):
            with open('image_summaries.pkl', 'rb') as f:
                image_summaries = pickle.load(f)
        else:
            st.warning("Please enter image summaries before searching.")
            return

        # Perform search based on the query
        vectorstore = Chroma(
            collection_name="mm_rag_gemini",
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        )

        retriever_multi_vector_img = create_multi_vector_retriever(
            vectorstore,
            text_summaries,
            texts,
            table_summaries,
            tables,
            image_summaries,
            img_base64_list
        )

        chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img, model_vision)
        
        search_results = chain_multimodal_rag.invoke(query)
        docs = retriever_multi_vector_img.get_relevant_documents(query, limit=1)
        print(split_image_text_types(docs))
        # Display search results
        if search_results:
            st.write(search_results)
            image_folder = "RAGFolder"  # Folder to save images
            os.makedirs(image_folder, exist_ok=True)  # Create the folder if it doesn't exist
            for index, image_data in enumerate((split_image_text_types(docs))['images']):
                filename = os.path.join(image_folder, f"image_{index}.jpg")
                base64_to_image(image_data, filename)
        else:
            st.write("No results found.")


        for image_file in os.listdir(image_folder):
            if image_file.endswith(".jpg"):  # Adjust extension if needed
                image_path = os.path.join(image_folder, image_file)
                st.image(image_path, caption=image_file)


if __name__ == "__main__":
    main()
