import src
from src.rag_engine import TrafficLawRAG
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def main():
    console.print(
        "🚦 [bold green]HỆ THỐNG CHATBOT LUẬT GIAO THÔNG (GEMINI RAG)[/bold green] 🚦"
    )
    console.print("-" * 50)

    try:
        bot = TrafficLawRAG()
    except Exception as e:
        console.print(f"❌ [red]Init Error:[/red] {e}")
        return

    console.print("\n✅ [bold blue]Ready! Type 'exit' to quit.[/bold blue]")

    while True:
        query = console.input("\n👤 [bold yellow]Bạn:[/bold yellow] ").strip()
        if query.lower() in ["exit", "quit", "thoát"]:
            break
        if not query:
            continue

        try:
            answer, sources = bot.chat(query)

            console.print(Panel(Markdown(answer), title="🤖 Bot", border_style="cyan"))

            console.print("\n📚 [bold magenta]Nguồn tham khảo:[/bold magenta]")
            for i, doc in enumerate(sources[:3]):
                citation = doc.metadata.get("citation", "N/A")
                console.print(f"   {i+1}. [italic]{citation}[/italic]")

        except Exception as e:
            console.print(f"❌ [red]Error:[/red] {e}")


if __name__ == "__main__":
    main()
