from langchain.prompts import ChatPromptTemplate

# Prompt biến đổi câu hỏi (Query Expansion)
QUERY_TRANSFORM_SYSTEM = """Bạn là chuyên gia pháp lý. Nhiệm vụ của bạn là viết lại câu hỏi của người dùng thành một câu truy vấn tìm kiếm chuẩn xác trong văn bản luật.
- Dùng từ ngữ chuyên ngành (Ví dụ: "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu giao thông").
- Giữ nguyên ý định tìm mức phạt hoặc hành vi.
- Chỉ trả về câu viết lại, không giải thích gì thêm."""

# Prompt trả lời câu hỏi (RAG Answer)
ANSWER_SYSTEM = """Bạn là Trợ lý Luật Giao thông AI.
Sử dụng thông tin sau để trả lời câu hỏi. 
- Trích dẫn chính xác (Nghị định, Điều, Khoản).
- Nếu không có thông tin, hãy nói không biết.

CONTEXT:
{context}
"""


def get_query_transform_prompt():
    return ChatPromptTemplate.from_messages(
        [("system", QUERY_TRANSFORM_SYSTEM), ("human", "Câu hỏi: {question}")]
    )


def get_answer_prompt():
    return ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM), ("human", "{question}")]
    )
