
import re
from typing import List, Dict
from langchain_core.documents import Document
from indexing.vector_store_utils import create_or_load_vector_store
from mappings.ipc_bns_mapping import mapper

class BNSRetriever:
    def __init__(self):
        self.vector_store, self.embeddings = create_or_load_vector_store()
        
    def _extract_ipc_sections(self, query: str) -> List[str]:
        # Regex to find "IPC" followed by number
        matches = re.findall(r"IPC\s*(\d+[A-Z]?)", query, re.IGNORECASE)
        return matches

    def retrieve(self, query: str, k: int = 12, score_threshold: float = 0.3) -> List[Document]:
        # 0. Normalize Query: Strip whitespace and common trailing punctuation
        # This addresses the user requirement: "Treat user queries the same regardless of punctuation"
        query = query.strip().rstrip(".,;!?")
        
        # 1. Check for IPC references and map to BNS
        ipc_refs = self._extract_ipc_sections(query)
        mapped_bns_sections = []
        
        for ipc_sec in ipc_refs:
            mapping = mapper.resolve_ipc(ipc_sec)
            if mapping:
                bns_sec = str(mapping["bns_section"])
                mapped_bns_sections.append(bns_sec)
                print(f"DEBUG: Found IPC {ipc_sec} -> Mapping to BNS {bns_sec}")
        
        if not self.vector_store:
            print("ERROR: Vector store not initialized.")
            return []

        # 2. Get semantic Search results
        # Use a larger k initially if we have mappings to filter
        search_k = k * 3 if mapped_bns_sections else k
        results = self.vector_store.similarity_search_with_score(query, k=search_k)
        
        final_docs = []
        seen_ids = set()
        
        # 3. Priority 1: Chunks explicitly mapped from IPC
        if mapped_bns_sections:
            for doc, distance in results:
                sec_num = str(doc.metadata.get("section_number", ""))
                if sec_num in mapped_bns_sections:
                    doc.metadata["score"] = 1.0 # Priority boost
                    doc.metadata["is_mapped"] = True
                    doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
                    if doc_id not in seen_ids:
                        final_docs.append(doc)
                        seen_ids.add(doc_id)

        # 4. Priority 2: Semantic matches
        for doc, distance in results:
            similarity = 1 / (1 + distance)
            doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
            
            if doc_id in seen_ids:
                continue
                
            # Strict Thresholding:
            # If IPC was mentioned (is_mapped=True), we are lenient because mappings are hard-coded rules.
            # If pure semantic search, we must be strict to avoid hallucinating unrelated sections.
            if distance > 1.1 and not doc.metadata.get("is_mapped", False):
                # distance is L2 distance (lower is better).
                # 0.0 = exact match. > 1.0 is very far.
                # Threshold ~0.65-0.7 keeps quality high for MiniLM.
                continue
            
            # Normalize score for UI (0 to 1)
            similarity = 1 / (1 + distance)
            doc.metadata["score"] = similarity
            
            final_docs.append(doc)
            seen_ids.add(doc_id)
            
        # 5. Sort by relevance and limit to k
        final_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
        return final_docs[:k]
