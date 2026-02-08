
from rag.retriever import BNSRetriever
try:
    r = BNSRetriever()
    print("--- Retrieval Rank Debug ---")
    query = "furnishing false information"
    results = r.vector_store.similarity_search_with_score(query, k=25)
    for i, (doc, score) in enumerate(results):
        sec = doc.metadata.get('section_number', 'N/A')
        print(f"Rank {i+1}: Section {sec} (Score: {score:.4f})")
        if sec == "212":
            print(f"   >>> FOUND TARGET SECTION 212 at Rank {i+1} <<<")
except Exception as e:
    print(e)
