import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re

load_dotenv()

# def list_markdown_files(root_dir):
#     md_files = []
#     for root, _, files in os.walk(root_dir):
#         for f in files:
#             if f.endswith(".md"):
#                 md_files.append(os.path.join(root, f))
#     return md_files

# root_path = "./docs_data/docs"
# markdown_files = list_markdown_files(root_path)

# for file in markdown_files:
#     process_doc_file(file)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
                    with open("./docs_data" + "/" + path, "r") as code_file:
                        code_content = code_file.read()
                    text_content += "\n```\n" + code_content + "\n```\n"
            else:
                text_content += "\n" + doc.page_content.strip() + "\n"

    return chunks

def vector_store(embedded_docs):
    faiss_path = './faiss_index'
    db = None

    if os.path.exists(faiss_path):
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

    if db:
        db.add_documents(embedded_docs)
    else:
        db = FAISS.from_documents(embedded_docs, embeddings)

    db.save_local(faiss_path)

def process_doc_file(file_path):
    loader = UnstructuredMarkdownLoader(
        file_path,
        mode="elements",
        strategy="fast",
    )

    docs = loader.load()
    chunks = create_chunks(docs)

    print(len(chunks))
    vector_store(chunks)

def reterieve_similar_documents(query, k=4):
    faiss_path = './faiss_index'
    if os.path.exists(faiss_path):
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        results = db.similarity_search(query, k=k)
        return results
    else:
        print("FAISS index not found.")
        return []

# process_doc_file("./docs_data/docs/tutorial/body-multiple-params.md")

docs = reterieve_similar_documents("How to use multiple parameters in a function?", k=4)
for d in docs:
    print("\n", d.page_content)
