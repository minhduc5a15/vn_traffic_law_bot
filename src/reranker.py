from sentence_transformers import CrossEncoder
from src.config import AppConfig


class Reranker:
    def __init__(self):
        print(
            f"⚖️  [Reranker] Loading: {AppConfig.RERANKER_MODEL} ({AppConfig.RERANKER_DEVICE})..."
        )
        self.model = CrossEncoder(
            AppConfig.RERANKER_MODEL, device=AppConfig.RERANKER_DEVICE
        )
        print("   -> ✅ Reranker Ready.")

    def rank_documents(self, query: str, documents: list):
        if not documents:
            return []

        inputs = [[query, doc.page_content] for doc in documents]

        # Predict scores
        scores = self.model.predict(
            inputs, batch_size=AppConfig.RERANKER_BATCH_SIZE, show_progress_bar=False
        )

        # Attach scores & Sort
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)

        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[: AppConfig.RERANK_TOP_K]]
