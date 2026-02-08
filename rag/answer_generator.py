
from config.settings import settings
from langchain_groq import ChatGroq
from rag.prompts import build_chat_prompt
from rag.retriever import BNSRetriever

class RAGController:
    def __init__(self):
        self.retriever = BNSRetriever()
        
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")
            
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0
        )

    def format_context(self, docs):
        context_str = ""
        for i, doc in enumerate(docs):
            sec_num = doc.metadata.get("section_number", "Unknown")
            title = doc.metadata.get("section_title", "")
            page_info = doc.metadata.get("page_range", doc.metadata.get("start_page", "Unknown"))
            text = doc.page_content.replace("\n", " ")
            context_str += f"[{i+1}] Section {sec_num}: {title} (Page: {page_info})\nTEXT: {text}\n\n"
        return context_str

    def answer_question(self, question: str):
        """Unified method for UI and CLI that returns answer + docs."""
        try:
            # 1. Retrieve
            docs = self.retriever.retrieve(question)
            
            # 2. Guard: No docs found
            if not docs:
                return {
                    "answer": "The requested information is not available in the official BNS document or the index is not ready.",
                    "documents": []
                }
            
            # 3. Format context
            context_text = self.format_context(docs)
            
            # 4. Build the full prompt
            from rag.prompts import BASE_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
            full_prompt = BASE_SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(
                context=context_text,
                question=question
            )
            
            # 5. Generate
            response = self.llm.invoke(full_prompt)
            
            return {
                "answer": response.content,
                "documents": docs
            }
        except Exception as e:
            print(f"CRITICAL ERROR in RAG Pipeline: {e}")
            return {
                "answer": f"An error occurred while processing your request: {str(e)}",
                "documents": []
            }

    def ask_question(self, question: str):
        """Legacy CLI support."""
        res = self.answer_question(question)
        return res["answer"]
