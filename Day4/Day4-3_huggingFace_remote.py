import dotenv
from langchain_huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                                   )
query_vector = embeddings.embed_query("What is the capital of France?")
print(query_vector)
print(len(query_vector))