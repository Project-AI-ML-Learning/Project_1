# chunker.py
# Think of this like SCISSORS.
# Job: take the big pages and cut them into
# small neat pieces that are easy to read.

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(pages):
    # Scissors setting:
    # cut every 1000 characters
    # but keep 200 characters from the previous cut
    # so we never lose a sentence at the edge
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Cut all pages into pieces
    chunks = splitter.split_documents(pages)
    return chunks