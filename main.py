import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# 2. Document Loading
pdf_path = os.path.join(os.path.dirname(__file__), "Maths competetive exam.pdf")
print(f"Loading document from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} document pages.")

# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

# 4. Embeddings & In-Memory Vector Store (Qdrant + FastEmbed)
print("Creating embeddings and indexing into Qdrant vector store...")
embeddings = FastEmbedEmbeddings()

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag_collection",
    embedding=embeddings,
)
vectorstore.add_documents(chunks)
print("Indexing complete!")

# 5. Initialize NVIDIA LLM
# Note: nemotron uses reasoning tokens, so max_completion_tokens should be >= 1024
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)

# 6. Retriever & RAG Prompt
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20},
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a patient, expert tutor in competitive mathematics. "
        "Explain relevant concepts and show clear, step-by-step reasoning. "
        "Answer using ONLY the provided context. If the context does not contain "
        "enough information to answer, say so rather than guessing.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])

rag_chain = prompt | llm | StrOutputParser()

def query_rag(question: str):
    print("\n" + "="*60)
    print(f"Query: {question}")
    print("="*60)
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{i}] (Page {page_num}): {preview}...")

    # Build context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    print("\nGenerating response from NVIDIA LLM...")
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:")
    print(response)
    print("="*60)
    return response

EXAMPLE_QUESTION = "What is the place value of 7 in 92547108?"


def interactive_cli():
    print("\nCompetitive Mathematics Tutor")
    print("Enter a question, or type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue
        query_rag(question)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask questions about the competitive mathematics PDF.")
    parser.add_argument(
        "--example",
        action="store_true",
        help="run the preserved example question instead of starting the interactive tutor",
    )
    args = parser.parse_args()

    if args.example:
        query_rag(EXAMPLE_QUESTION)
    else:
        interactive_cli()