
from rag.retriever import BNSRetriever

r = BNSRetriever()
query = "furnishing false information"
print(f"Query: {query}")
print("-" * 30)

# Check raw search scores
results = r.vector_store.similarity_search_with_score(query, k=20)

print(f"{'Rank':<5} | {'Score':<10} | {'Section':<10} | {'Preview'}")
for i, (doc, score) in enumerate(results):
    sec = doc.metadata.get('section_number', 'N/A')
    preview = doc.page_content[:40].replace('\n', ' ')
    print(f"{i+1:<5} | {score:.4f}     | {sec:<10} | {preview}")
