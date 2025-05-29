from langchain_community.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain_google_vertexai import VertexAIEmbeddings
from config import CONNECTION_STRING, COLLECTION_NAME
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\LucasAlvesRibeiro\\Downloads\\jusec-chatbot-4fca0b96d3eb.json"


embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")

db = PGVector(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    distance_strategy=DistanceStrategy.COSINE,
    use_jsonb=True
)
