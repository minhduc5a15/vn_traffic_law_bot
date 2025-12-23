import os
import json
import csv
from typing import List
from langchain_core.documents import Document

# --- FIX LỖI TELEMETRY CHROMADB ---
# Tắt tính năng gửi thống kê của ChromaDB để tránh lỗi "capture() takes 1..."
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from src.ingestion import VietnameseLawParser
from src.indexing import Indexer


def save_data_to_debug(documents: List[Document], output_folder="./data/processed"):
    """
    Hàm lưu dữ liệu đã parse ra file JSON và CSV để con người kiểm tra.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Lưu dạng JSON (Dễ dùng cho code khác nếu cần)
    json_path = os.path.join(output_folder, "processed_chunks.json")
    data_export = []
    for doc in documents:
        data_export.append({"content": doc.page_content, "metadata": doc.metadata})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_export, f, ensure_ascii=False, indent=4)
    print(f"   -> 💾 Đã lưu file JSON kiểm tra tại: {json_path}")

    # 2. Lưu dạng CSV (Dễ mở bằng Excel/Google Sheet để soi lỗi)
    csv_path = os.path.join(output_folder, "processed_chunks.csv")
    with open(
        csv_path, "w", newline="", encoding="utf-8-sig"
    ) as f:  # utf-8-sig để Excel mở không lỗi font
        writer = csv.writer(f)
        # Header
        writer.writerow(
            ["Source File", "Article ID", "Content Preview", "Full Content"]
        )

        for doc in documents:
            source = doc.metadata.get("source", "")
            art_id = doc.metadata.get("article_id", "")
            content = doc.page_content
            # Lưu preview 100 ký tự đầu, và full content
            writer.writerow([source, art_id, content[:100].replace("\n", " "), content])

    print(f"   -> 💾 Đã lưu file CSV kiểm tra tại: {csv_path}")


def main():
    # --- GIAI ĐOẠN 1: INGESTION ---
    print("🚀 BẮT ĐẦU QUY TRÌNH ETL (Extract - Transform - Load)...")
    data_folder = "./data/raw"

    parser = VietnameseLawParser(data_folder)
    docs = parser.load_and_parse()

    if not docs:
        print("❌ Không có dữ liệu. Dừng chương trình.")
        return

    # --- BƯỚC PHỤ: LƯU DATA KIỂM TRA ---
    print("\n🧐 Đang xuất dữ liệu ra thư mục 'debug_data' để kiểm tra...")
    save_data_to_debug(docs)

    # --- GIAI ĐOẠN 2: INDEXING ---
    print("\n🏗️ BẮT ĐẦU GIAI ĐOẠN INDEXING...")
    indexer = Indexer()

    # Thực hiện build index
    try:
        # Lưu ý: Indexer sẽ tự xử lý việc xóa DB cũ nếu cần (như logic đã viết trong src/indexing.py)
        indexer.build_indices(docs)

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình Indexing: {e}")


if __name__ == "__main__":
    main()
