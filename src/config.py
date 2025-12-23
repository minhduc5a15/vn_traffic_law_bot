import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()


class AppConfig:
    # --- PATHS (Đường dẫn) ---
    # Lấy thư mục gốc của dự án
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
    DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

    VECTOR_DB_DIR = os.path.join(ROOT_DIR, "data", "indexes", "chroma_db")
    BM25_PATH = os.path.join(ROOT_DIR, "data", "indexes", "bm25_retriever.pkl")

    # --- MODELS (Cấu hình Model) ---
    EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
    EMBEDDING_DEVICE = "cuda"

    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE = "cuda"

    LLM_MODEL_NAME = "gemini-2.5-flash"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # --- PARAMETERS (Tham số thuật toán) ---
    RETRIEVAL_BM25_K = 15  # Số lượng docs lấy từ BM25
    RETRIEVAL_VECTOR_K = 40  # Số lượng docs lấy từ Vector Search
    RERANK_TOP_K = 5  # Số lượng docs cuối cùng sau khi chấm điểm lại

    RERANKER_BATCH_SIZE = 8


# Tự động tạo các thư mục cần thiết
os.makedirs(AppConfig.DATA_RAW_DIR, exist_ok=True)
os.makedirs(AppConfig.DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(AppConfig.VECTOR_DB_DIR), exist_ok=True)
