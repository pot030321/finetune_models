import os
from dotenv import load_dotenv

from typing_extensions import TypedDict
from operator import itemgetter

from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores.qdrant import Qdrant

from app.core.database.qdrant import Qdrant as QdrantDB
from app.integrations.gemini import GeminiEmbeddings

# ==================== Load env ====================
load_dotenv(".env")

# ==================== Qdrant Setup ====================
qdrant = QdrantDB()
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "documents")

embedding = GeminiEmbeddings(
    model="models/text-embedding-004"
)

vectordb = Qdrant(
    client=qdrant.client,
    collection_name=qdrant_collection_name,
    embeddings=embedding,
)

# ==================== Prompt + LLM ====================
template = """
Answer the following question as accurately as possible using the provided context:

Context:
{context}

Question:
{question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.environ["GEMINI_API_KEY"]
)

# ==================== RAG Pipeline ====================
class RagInput(TypedDict):
    question: str

final_chain = (
    RunnableParallel(
        context=(itemgetter("question") | vectordb.as_retriever()),
        question=itemgetter("question"),
    )
    | RunnableParallel(
        answer=(ANSWER_PROMPT | llm),
        docs=itemgetter("context")
    )
).with_types(input_type=RagInput)
