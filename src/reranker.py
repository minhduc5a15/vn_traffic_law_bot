from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.device = "cpu"
        print(
            f"⚖️  Đang tải Cross-Encoder (SOTA Multilingual): {model_name} trên {self.device.upper()}..."
        )
        print(
            "   ⚠️ Lưu ý: Chạy trên CPU sẽ chậm hơn một chút nhưng đảm bảo không bị Crash."
        )

        self.model = CrossEncoder(model_name, device=self.device)
        print("   -> ✅ Reranker đã sẵn sàng (CPU Mode).")

    def rank_documents(self, query: str, documents: list, top_k=5):
        if not documents:
            return []

        inputs = [[query, doc.page_content] for doc in documents]

        scores = self.model.predict(inputs, batch_size=8, show_progress_bar=False)

        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)

        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]
