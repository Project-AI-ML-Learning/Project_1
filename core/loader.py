# loader.py
# Think of this like a POSTMAN.
# Job: go pick up the PDF and bring the pages home.

from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(filepath):
    # Postman goes to the address (filepath)
    # picks up the document
    # brings back all the pages
    loader = PyPDFLoader(filepath)
    pages  = loader.load()
    return pages


def load_all_pdfs(folder):
    # Postman checks the entire street (folder)
    # picks up ALL PDFs one by one
    # returns everything in one big bag
    all_pages = []

    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder, filename)
            pages    = load_pdf(filepath)

            # Stamp which house it came from
            for page in pages:
                page.metadata["source_file"] = filename

            all_pages.extend(pages)

    return all_pages

"""Postman goes street by street. Every letter (page) gets a stamp saying which house it came from. Returns the full mailbag."""