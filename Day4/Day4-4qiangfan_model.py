import dotenv
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()
embeddings = QianfanEmbeddingsEndpoint()

query_vector = embeddings.embed_query("What is the capital of France?")
print(query_vector)
print(len(query_vector))