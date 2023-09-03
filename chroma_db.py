import chromadb 
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name = "collection")
collection.add(
    documents = ["Harry Potter and the half blood prince", "Harry Potter and the goblet of fire"],
    metadatas = [{"source": "half blood prince"}, {"source":"half blood prince"}],
    ids = ["id1", "id2"]
)


results = collection.query(
    query_texts = ["what is in this document"],
    n_results = 2
)

print(results)