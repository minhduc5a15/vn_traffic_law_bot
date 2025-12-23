import os
import pickle
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Thay đổi thư viện import
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Load biến môi trường
load_dotenv()


class Indexer:

    def __init__(
        self,
        db_path="./data/indexes/chroma_db",
        bm25_path="./data/indexes/bm25_retriever.pkl",
    ):
        self.db_path = db_path
        self.bm25_path = bm25_path

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️  Thiết bị chạy Embedding: {device.upper()}")

        # Sử dụng model tiếng Việt chuyên dụng từ BKAI hoặc VinAI
        # model_name = "bkai-foundation-models/vietnamese-bi-encoder"
        model_name = "bkai-foundation-models/vietnamese-bi-encoder"

        print(f"📥 Đang tải/Load model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True
            },  # Quan trọng cho so sánh cosine similarity
        )

    def build_indices(self, documents: list[Document]):
        print(f"📊 Đang tạo Index cho {len(documents)} documents...")

        # 1. Xây dựng Vector Store (ChromaDB)
        print("   -> 🧠 Đang embedding và lưu vào ChromaDB (Local sBERT)...")

        # Xóa DB cũ để clean
        if os.path.exists(self.db_path):
            import shutil

            shutil.rmtree(self.db_path)

        # Chroma tự động gọi model embedding để vector hóa documents
        # Batch size mặc định có thể chỉnh nếu bị OOM (Out of Memory)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            collection_metadata={"hnsw:space": "cosine"},  # Dùng Cosine Similarity
        )
        print("   -> ✅ Đã lưu Vector Index.")

        # 2. Xây dựng BM25 Retriever (Keyword Search)
        print("   -> 🔍 Đang tạo chỉ mục BM25 (Keyword Search)...")

        # Tokenizer cơ bản cho BM25 (tách từ theo khoảng trắng là tạm ổn cho BM25 ở bước này,
        # hoặc dùng pyvi nếu muốn chính xác hơn, nhưng mặc định vẫn hoạt động tốt)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10  # Lấy rộng ra một chút cho BM25

        with open(self.bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print("   -> ✅ Đã lưu BM25 Index.")

        return vectorstore, bm25_retriever

    def load_indices(self):
        """Hàm load lại index"""
        print("📂 Đang tải lại Index từ đĩa...")

        vectorstore = Chroma(
            persist_directory=self.db_path, embedding_function=self.embeddings
        )

        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)
        else:
            raise FileNotFoundError("Chưa tìm thấy file BM25 index.")

        return vectorstore, bm25_retriever
