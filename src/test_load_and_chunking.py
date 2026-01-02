from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import re

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_anchor_links(text):
    match = re.search(r'\{\s*#([A-Za-z0-9\-_]+)\s*\}', text)
    anchor = ""
    if match:
        anchor = match.group(1)
        
    return anchor

def create_chunks(docs):
    text_content = ""
    chunks = []

    for doc in docs:
        if doc.metadata["category"] == "Title":
            if not text_content:
                text_content += doc.page_content.strip() + "\n"
                anchor_link_text = extract_anchor_links(doc.page_content.strip())
            else:
                chunks.append(Document(page_content=text_content.strip(), metadata={"source": doc.metadata.get("source", "")[:-3] + (f"#{anchor_link_text}" if anchor_link_text else "")}))
                text_content = doc.page_content.strip() + "\n"
                anchor_link_text = extract_anchor_links(doc.page_content.strip())
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

def process_doc_file(file_path):
    loader = UnstructuredMarkdownLoader(
        file_path,
        mode="elements",
        strategy="fast",
    )

    docs = loader.load()

    for chunk in create_chunks(docs):
        print(chunk)

process_doc_file("./docs_data/docs/tutorial/body-multiple-params.md")