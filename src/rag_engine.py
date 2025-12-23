import pickle
import os
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import AppConfig
from src.prompts import get_answer_prompt, get_query_transform_prompt
from src.reranker import Reranker


class TrafficLawRAG:
    def __init__(self):
        print(f"🚀 [RAG Engine] Starting... (LLM: {AppConfig.LLM_MODEL_NAME})")

        # 1. Embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL,
            model_kwargs={"device": AppConfig.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2. Vector DB
        self.vector_db = Chroma(
            persist_directory=AppConfig.VECTOR_DB_DIR,
            embedding_function=self.embedding_model,
        )

        # 3. BM25
        with open(AppConfig.BM25_PATH, "rb") as f:
            self.bm25_retriever = pickle.load(f)
        self.bm25_retriever.k = AppConfig.RETRIEVAL_BM25_K

        # 4. Reranker
        self.reranker = Reranker()

        # 5. LLM (Gemini)
        if not AppConfig.GOOGLE_API_KEY:
            raise ValueError("❌ Missing GOOGLE_API_KEY in .env")

        self.llm = ChatGoogleGenerativeAI(
            model=AppConfig.LLM_MODEL_NAME,
            temperature=0,
            api_key=AppConfig.GOOGLE_API_KEY,
        )

        # 6. Prompts
        self.answer_prompt = get_answer_prompt()
        self.query_transform_prompt = get_query_transform_prompt()

    def generate_legal_query(self, user_query: str):
        print(f"   🔄 Normalizing query: '{user_query}'")
        try:
            response = (self.query_transform_prompt | self.llm).invoke(
                {"question": user_query}
            )
            legal_query = response.content.strip()
            print(f"   -> 🎯 Legal Query: '{legal_query}'")
            return legal_query
        except Exception as e:
            print(f"   ⚠️ Error expanding query: {e}. Using original.")
            return user_query

    def retrieve_hybrid(self, query: str):
        # Step 1: Query Expansion
        search_query = self.generate_legal_query(query)

        # Step 2: Retrieval
        docs_vector = self.vector_db.similarity_search(
            search_query, k=AppConfig.RETRIEVAL_VECTOR_K
        )
        docs_bm25 = self.bm25_retriever.invoke(search_query)

        # Step 3: Deduplication
        unique_docs = {}
        for doc in docs_vector + docs_bm25:
            # Dùng citation làm key để lọc trùng
            key = doc.metadata.get("citation", doc.page_content[:50])
            unique_docs[key] = doc

        merged_docs = list(unique_docs.values())
        print(f"   -> Found {len(merged_docs)} potential candidates.")

        # Step 4: Reranking
        print("   -> ⚖️  Reranking...")
        final_docs = self.reranker.rank_documents(query, merged_docs)
        return final_docs

    def chat(self, user_query: str):
        context_docs = self.retrieve_hybrid(user_query)

        if not context_docs:
            return "Xin lỗi, không tìm thấy tài liệu liên quan.", []

        # Format Context
        context_text = ""
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get("citation", "N/A")
            content = doc.page_content.replace("\n", " ")
            context_text += f"[{i+1}] {source}: {content}\n\n"

        # Generation
        chain = self.answer_prompt | self.llm
        response = chain.invoke({"context": context_text, "question": user_query})

        return response.content, context_docs
