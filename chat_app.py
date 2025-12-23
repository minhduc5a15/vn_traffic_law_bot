import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
from dotenv import load_dotenv
from src.rag_engine import TrafficLawRAG


# Load API Key
load_dotenv()


def main():
    print("🚦 HỆ THỐNG CHATBOT LUẬT GIAO THÔNG (HYBRID RAG) 🚦")
    print("-" * 50)

    try:
        # Khởi tạo Engine (Load model mất khoảng 5-10s)
        bot = TrafficLawRAG()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return

    print("\n✅ Hệ thống đã sẵn sàng! Gõ 'exit' để thoát.")

    while True:
        query = input("\n👤 Bạn: ")
        if query.lower() in ["exit", "quit", "thoát"]:
            break

        if not query.strip():
            continue

        try:
            # Gọi hàm chat
            answer, sources = bot.chat(query)

            print(f"\n🤖 Bot: {answer}")

            # Hiển thị nguồn trích dẫn (Evidence)
            print("\n📚 Nguồn tham khảo (Top 3 Reranked):")
            for i, doc in enumerate(sources[:3]):
                score = doc.metadata.get("rerank_score", 0.0)
                citation = doc.metadata.get("citation", "N/A")
                print(f"   {i+1}. {citation} (Độ phù hợp: {score:.4f})")

        except Exception as e:
            print(f"❌ Lỗi xử lý: {e}")


if __name__ == "__main__":
    main()
