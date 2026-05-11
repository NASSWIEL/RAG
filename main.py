"""Entry point for the RAG (Retrieval-Augmented Generation) interactive query demo."""

from rag_engine import RAGEngine


def main():
    """Run the interactive RAG query loop using a remote PDF as the knowledge source."""
    pdf_url = "https://arxiv.org/pdf/2005.11401.pdf"

    rag = RAGEngine(pdf_url)

    print("\n\nInteractive mode - Enter your questions (type 'q' to exit):")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["q"]:
            break
        answer = rag.query(user_query)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
