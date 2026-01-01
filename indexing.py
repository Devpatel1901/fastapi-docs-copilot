import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def list_markdown_files(root_dir):
    md_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))
    return md_files

def create_chunks(docs):
    text_content = ""
    chunks = []

    for doc in docs:
        if doc.metadata["category"] == "Title":
            if not text_content:
                text_content += doc.page_content + "\n"
            else:
                chunks.append(Document(page_content=text_content.strip(), metadata={"source": doc.metadata.get("source", "")}))
                text_content = doc.page_content.strip() + "\n"
        else:
            if doc.metadata.get("emphasized_text_tags", []) == ['i']:
                match = re.search(r"\{\s*(?:\.\./)*([^\s]+\.py)", doc.page_content)
                if match:
                    path = match.group(1)
                    try:
                        with open("./docs_data" + "/" + path, "r") as code_file:
                            code_content = code_file.read()
                        text_content += "\n```\n" + code_content + "\n```\n"
                    except FileNotFoundError:
                        logging.warning(f"Code file not found: {path}")
                    except Exception as e:
                        logging.error(f"Error reading code file {path}: {e}")
            else:
                text_content += "\n" + doc.page_content.strip() + "\n"

    return chunks

def vector_store(embedded_docs):
    faiss_path = './faiss_index'
    db = None

    if os.path.exists(faiss_path):
        logging.debug(f"Loading existing FAISS index from {faiss_path}")
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

    if db:
        logging.debug("Adding documents to existing FAISS index")
        db.add_documents(embedded_docs)
    else:
        logging.debug("Creating new FAISS index")
        db = FAISS.from_documents(embedded_docs, embeddings)

    logging.debug(f"Saving FAISS index to {faiss_path}")
    db.save_local(faiss_path)

def initiate_indexing_process(dir_path):
    logging.info(f"Listing markdown files in directory: {dir_path}")
    file_list = list_markdown_files(dir_path)

    logging.info(f"Found {len(file_list)} markdown files to process.")

    for file in file_list:
        logging.info(f"Start processing for file: {file}")

        loader = UnstructuredMarkdownLoader(
            file,
            mode="elements",
            strategy="fast",
        )

        docs = loader.load()
        chunks = create_chunks(docs)

        logging.info(f"Created {len(chunks)} chunks from file: {file}")

        if chunks:
            vector_store(chunks)
        else:
            logging.info(f"No chunks created from file: {file}, skipping vector store update.")

        logging.info(f"Completed processing for file: {file}")

if __name__ == "__main__":
    logging.info("Starting the indexing process...")
    initiate_indexing_process("./docs_data/docs")