import sys
import json
import csv
import os
import src
from src.ingestion import VietnameseLawParser
from src.indexing import Indexer
from src.config import AppConfig


def save_debug_data(documents):
    json_path = os.path.join(AppConfig.DATA_PROCESSED_DIR, "processed_chunks.json")
    csv_path = os.path.join(AppConfig.DATA_PROCESSED_DIR, "processed_chunks.csv")

    data_export = [
        {"content": d.page_content, "metadata": d.metadata} for d in documents
    ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_export, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Article ID", "Citation", "Content"])
        for doc in documents:
            writer.writerow(
                [
                    doc.metadata.get("article", ""),
                    doc.metadata.get("citation", ""),
                    doc.page_content,
                ]
            )

    print(f"   -> 💾 Debug data saved to {AppConfig.DATA_PROCESSED_DIR}")


def main():
    print("🚀 STARTING ETL PIPELINE")
    print("-" * 50)

    # 1. Ingestion
    parser = VietnameseLawParser()
    docs = parser.load_and_parse()

    if not docs:
        print("❌ No documents found. Check 'data/raw' folder.")
        return

    save_debug_data(docs)

    # 2. Indexing
    print("\n🏗️  STARTING INDEXING")
    indexer = Indexer()
    try:
        indexer.build_indices(docs)
        print("\n🎉 SETUP COMPLETE! Run 'python chat_app.py' to start.")
    except Exception as e:
        print(f"\n❌ Indexing Error: {e}")


if __name__ == "__main__":
    main()
