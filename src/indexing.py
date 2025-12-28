import os
import pickle
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from src.config import AppConfig


class Indexer:
    def __init__(self):
        self.db_path = AppConfig.VECTOR_DB_DIR
        self.bm25_path = AppConfig.BM25_PATH

        print(
            f"⚙️  [Indexer] Init Embedding Model: {AppConfig.EMBEDDING_MODEL} ({AppConfig.EMBEDDING_DEVICE})"
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL,
            model_kwargs={"device": AppConfig.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

    def build_indices(self, documents: List[Document]):
        print(f"📊 Đang tạo Index cho {len(documents)} documents...")

        # 1. Vector Store
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        print("   -> 🧠 Embedding & ChromaDB...")
        Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print("   -> ✅ Vector Index Saved.")

        # 2. BM25
        print("   -> 🔍 Creating BM25 Index...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = AppConfig.RETRIEVAL_BM25_K

        with open(self.bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print("   -> ✅ BM25 Index Saved.")
