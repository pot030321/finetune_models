import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import backoff
import qdrant_client
import uuid6

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.schema import Document
from openai.types import Embedding
from qdrant_client.http.models import PointStruct
from concurrent.futures import ThreadPoolExecutor, as_completed
from httpx import ReadTimeout, WriteTimeout

from app.core.database.qdrant import Qdrant
from app.integrations.gemini import GeminiProvider

load_dotenv(".env")

qdrant = Qdrant()
geminiProvider = GeminiProvider()

qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "documents")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunks(lst, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_file(file_path: str):
    """Process a single file and add it to Qdrant"""
    try:
        filetype = file_path.lower().split(".")[-1]
        logger.info(f"Processing file: {file_path} of type: {filetype}")

        # Load document based on file type
        if filetype == "json":
            docs = process_json_file(file_path)
        elif filetype == "pdf":
            loader = PyMuPDFLoader(file_path=file_path)
            docs = loader.load()
        elif filetype in ["txt"]:
            loader = TextLoader(file_path=file_path, encoding="UTF-8")
            docs = loader.load()
        else:
            logger.info(f"Unsupported file type: {filetype}")
            return

        # Split documents
        splits = split_documents(docs, filetype)

        # Create collection name from filename
        # collection_name = os.path.splitext(os.path.basename(file_path))[0]
        collection_name = qdrant_collection_name

        # Create collection in Qdrant
        qdrant.create_collection(collection_name=collection_name)

        # Process splits in chunks
        process_splits(splits, collection_name, file_path)

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")


def process_json_file(file_path: str) -> List[Document]:
    """Process JSON file with markdown content"""

    def metadata_func(record: dict, metadata: dict) -> dict:
        if "metadata" in record:
            metadata.update(record["metadata"])
        metadata["source"] = file_path
        return metadata

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        content_key="markdown",
        text_content=True,
        metadata_func=metadata_func,
    )

    try:
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading JSON with array schema: {str(e)}")
        # Fallback to non-array schema
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".",
            content_key="markdown",
            text_content=True,
            metadata_func=metadata_func,
        )
        return loader.load()


def split_documents(docs: List[Document], filetype: str) -> List[Document]:
    """Split documents based on file type"""
    if filetype == "json":
        return split_markdown_documents(docs)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300
        )
        return text_splitter.split_documents(docs)


def split_markdown_documents(docs: List[Document]) -> List[Document]:
    """Split markdown documents using header-based splitting"""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
        return_each_line=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )

    splits = []
    for doc in docs:
        try:
            if len(doc.page_content) > 6000:
                initial_splits = text_splitter.split_text(doc.page_content)
            else:
                initial_splits = [doc.page_content]

            for text in initial_splits:
                try:
                    md_splits = markdown_splitter.split_text(text)
                    if md_splits:
                        for split in md_splits:
                            combined_metadata = {**doc.metadata, **split.metadata}
                            splits.append(
                                Document(
                                    page_content=split.page_content,
                                    metadata=combined_metadata,
                                )
                            )
                    else:
                        # Fallback to character splitting
                        char_splits = text_splitter.split_text(text)
                        for char_split in char_splits:
                            splits.append(
                                Document(page_content=char_split, metadata=doc.metadata)
                            )
                except Exception as e:
                    logger.warning(f"Falling back to character splitting: {str(e)}")
                    char_splits = text_splitter.split_text(text)
                    for char_split in char_splits:
                        splits.append(
                            Document(page_content=char_split, metadata=doc.metadata)
                        )
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            continue

    return splits


def process_splits(splits: List[Document], collection_name: str, file_path: str):
    """Process and add splits to Qdrant"""
    BATCH_SIZE = 200
    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i : i + BATCH_SIZE]
        try:
            embeddings = create_embeddings(batch)
            create_points(batch, embeddings, file_path)
            logger.info(
                f"Added batch {i//BATCH_SIZE + 1} to collection {collection_name}"
            )
        except Exception as e:
            logger.error(f"Error processing batch {i//BATCH_SIZE + 1}: {str(e)}")


def create_embeddings(
    docs: List[Document], batch_size: int = 100, max_workers: int = 5
) -> List[Embedding]:
    """
    Create embeddings for documents with optimized batching and parallel processing

    Args:
        docs: List of documents to embed
        batch_size: Number of texts to process in each batch
        max_workers: Maximum number of parallel workers

    Returns:
        List of embeddings
    """
    texts = [doc.page_content for doc in docs]
    batches = list(chunks(texts, batch_size))
    all_embeddings = []

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(batch_embeddings_create, batch): i
            for i, batch in enumerate(batches)
        }

        # Collect results while maintaining order
        batch_results = [None] * len(batches)
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                result = future.result()
                batch_results[batch_idx] = result
            except Exception as e:
                logging.error(f"Batch {batch_idx} failed: {str(e)}")
                raise

    # Flatten results while maintaining order
    for batch in batch_results:
        if batch:
            all_embeddings.extend(batch)

    return all_embeddings


def batch_embeddings_create(
    texts_batch: List[str], retry_count: int = 5
) -> List[Dict[str, Any]]:
    """Helper function to create embeddings for a batch of texts with retries"""

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=retry_count,
        giveup=lambda e: "rate_limit_exceeded" not in str(e).lower(),
    )
    def _create_embeddings(batch: List[str]) -> List[Dict[str, Any]]:
        return geminiProvider.embeddings_create(input=batch).data

    try:
        return _create_embeddings(texts_batch)
    except Exception as e:
        logging.error(
            f"Failed to create embeddings for batch after {retry_count} retries: {str(e)}"
        )
        raise


def create_points(
    docs: List[Document],
    embeds: List[Embedding],
    key: str,
):
    collection_name = qdrant_collection_name
    # Point structure
    points = [
        PointStruct(
            id=str(uuid6.uuid7()),
            vector=data.embedding,
            payload={
                "key": key,
                "metadata": doc.metadata,
                "page_content": doc.page_content,
            },
        )
        for idx, (data, doc) in enumerate(zip(embeds, docs))
    ]

    # Split points into smaller chunks for processing
    point_chunks = list(chunks(points, 50))  # Process 50 points at a time

    logging.info(
        f"Processing {len(point_chunks)} chunks for collection {collection_name}"
    )

    for chunk_idx, chunk in enumerate(point_chunks):

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=7,
            max_time=300,
            giveup=lambda e: not (
                isinstance(
                    e,
                    (
                        qdrant_client.http.exceptions.ResponseHandlingException,
                        ReadTimeout,
                        WriteTimeout,
                    ),
                )
                or "timeout" in str(e).lower()
            ),
            on_backoff=lambda details: logging.info(
                f"Backing off {details['wait']}s after {details['tries']} tries. "
                f"Exception: {details['exception'].__class__.__name__}: {str(details['exception'])}"
            ),
        )
        def add_points_with_retry():
            try:
                logging.info(
                    f"Attempting to add chunk {chunk_idx + 1}/{len(point_chunks)} "
                    f"({len(chunk)} points) to collection {collection_name}"
                )

                # Remove timeout parameter as it's not supported
                result = qdrant.add_points(
                    collection_name=collection_name, points=chunk, wait=True
                )

                logging.info(
                    f"Successfully added chunk {chunk_idx + 1}/{len(point_chunks)} "
                    f"to collection {collection_name}. Response: {result}"
                )
                return result

            except Exception as e:
                logging.error(
                    f"Failed to add chunk {chunk_idx + 1}/{len(point_chunks)} "
                    f"to collection {collection_name}. Error details: "
                    f"{e.__class__.__name__}: {str(e)}"
                )
                if hasattr(e, "response"):
                    logging.error(
                        f"Server response: {e.response.text if hasattr(e.response, 'text') else e.response}"
                    )
                raise

        try:
            add_points_with_retry()
        except Exception as e:
            logging.error(
                f"Failed to add points chunk {chunk_idx + 1}/{len(point_chunks)} "
                f"to collection {collection_name} after all retries: {str(e)}"
            )
            raise


def main():
    source_docs_path = os.path.abspath("./source_docs")

    for root, _, files in os.walk(source_docs_path):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path)


if __name__ == "__main__":
    main()
