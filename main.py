from rag_engine import RAGEngine

def main():
    pdf_url = "https://arxiv.org/pdf/2005.11401.pdf"
    
    rag = RAGEngine(pdf_url)
    
    
    print("\n\nInteractive mode - Enter your questions (type 'q' to exit):")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ['q']:
            break
        answer = rag.query(user_query)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
