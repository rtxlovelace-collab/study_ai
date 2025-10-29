from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings",
                                   cache_folder="./embeddings/")
query_vector = embeddings.embed_query("What is the capital of France?")
print(query_vector)
print(len(query_vector))