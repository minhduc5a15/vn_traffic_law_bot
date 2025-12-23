import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

from src.reranker import Reranker


class TrafficLawRAG:

    def __init__(
        self,
        vector_db_path="./data/indexes/chroma_db",
        bm25_path="./data/indexes/bm25_retriever.pkl",
    ):
        print("🚀 Khởi động Traffic Law RAG Engine (v2.0 - Query Expansion)...")

        # 1. Load Embeddings
        device = "cpu"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="bkai-foundation-models/vietnamese-bi-encoder",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2. Vector DB
        self.vector_db = Chroma(
            persist_directory=vector_db_path, embedding_function=self.embedding_model
        )

        # 3. BM25
        with open(bm25_path, "rb") as f:
            self.bm25_retriever = pickle.load(f)
        self.bm25_retriever.k = 15  # Lấy top 15 BM25

        # 4. Reranker
        self.reranker = Reranker()

        # 5. LLM Chính & LLM Query Gen
        api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0, api_key=api_key
        )

        # Prompt
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Bạn là Trợ lý Luật Giao thông AI.
            Sử dụng thông tin sau để trả lời câu hỏi. 
            - Trích dẫn chính xác (Nghị định, Điều, Khoản).
            - Nếu không có thông tin, hãy nói không biết.
            
            CONTEXT:
            {context}
            """,
                ),
                ("human", "{question}"),
            ]
        )

        # Prompt biến đổi câu hỏi
        self.query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Bạn là chuyên gia pháp lý. Nhiệm vụ của bạn là viết lại câu hỏi của người dùng thành một câu truy vấn tìm kiếm chuẩn xác trong văn bản luật.
            - Dùng từ ngữ chuyên ngành (Ví dụ: "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu giao thông").
            - Giữ nguyên ý định tìm mức phạt hoặc hành vi.
            - Chỉ trả về câu viết lại, không giải thích gì thêm.""",
                ),
                ("human", "Câu hỏi: {question}"),
            ]
        )

    def generate_legal_query(self, user_query: str):
        print(f"   🔄 Đang chuẩn hóa câu hỏi: '{user_query}'")
        response = (self.query_transform_prompt | self.llm).invoke(
            {"question": user_query}
        )
        legal_query = response.content.strip()
        print(f"   -> 🎯 Query Luật: '{legal_query}'")
        return legal_query

    def retrieve_hybrid(self, query: str, top_k_final=5):
        search_query = self.generate_legal_query(query)

        docs_vector = self.vector_db.similarity_search(search_query, k=40)
        docs_bm25 = self.bm25_retriever.invoke(search_query)

        unique_docs = {}
        for doc in docs_vector + docs_bm25:
            key = doc.metadata.get("citation", doc.page_content[:50])
            unique_docs[key] = doc
        merged_docs = list(unique_docs.values())

        print(f"   -> Tìm thấy {len(merged_docs)} tài liệu tiềm năng.")

        print("   -> ⚖️  Reranking...")
        final_docs = self.reranker.rank_documents(query, merged_docs, top_k=top_k_final)

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
