import ollama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import time
import sys

# --- Configuration ---
# Ollama models for LLM (Llama 3) and Embeddings (nomic-embed-text)
LLM_MODEL = 'llama3'
EMBEDDING_MODEL = 'nomic-embed-text'
VECTOR_DB_PATH = "./chroma_db"
DOCUMENT_FILE = "financial_data.md" # The file created in Step 2.A

def setup_rag_pipeline():
    """Sets up the RAG components: Loader, Splitter, Embeddings, and Vector Store."""
    
    print("--- Phase 1: Data Preparation ---")
    
    # 1. Load the document
    print(f"Loading document: {DOCUMENT_FILE}")
    loader = TextLoader(DOCUMENT_FILE)
    docs = loader.load()

    # 2. Split the document into chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")
    
    # Check if the embedding model is pulled (Ollama requires separate pull for embeddings)
    try:
        ollama.pull(EMBEDDING_MODEL)
    except Exception as e:
        print(f"Warning: Could not automatically pull {EMBEDDING_MODEL}. Please run 'ollama pull {EMBEDDING_MODEL}' manually.")
        
    # 3. Create Ollama Embeddings object
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 4. Create the Vector Store (ChromaDB)
    print(f"Creating/Loading Vector Store in: {VECTOR_DB_PATH}")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=VECTOR_DB_PATH
    )

    # 5. Create the Retriever (which fetches the top chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 6. Define the LLM and the Prompt Template
    llm = Ollama(model=LLM_MODEL)
    
    # The system prompt enforces the persona and the RAG rule (use context)
    system_prompt = (
        "You are a helpful, unbiased, and compliant financial service agent. "
        "Your task is to answer the user's question based ONLY on the provided context, "
        "which is a section of the ABC Bank Terms of Service and Loan FAQ. "
        "If the context does not contain the answer, state clearly that you cannot find the relevant information. "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    # 7. Create the RAG Chain
    # The chain combines retrieval (getting chunks) and generation (LLM response)
    rag_chain = prompt | llm

    return retriever, rag_chain

def run_rag_chat(retriever, rag_chain):
    """Main loop for the RAG agent."""
    
    print("\n--- RAG Agent Console Ready ---")
    print(f"Using {LLM_MODEL} grounded by {DOCUMENT_FILE}.")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 35)

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', '/bye']:
            print("\nSession ended. Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            start_time = time.time()
            
            # 1. RETRIEVAL: Get the relevant documents based on the user's query
            retrieved_docs = retriever.invoke(user_input)
            
            # 2. PREPARATION: Format the retrieved context for the LLM prompt
            context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
            
            # 3. GENERATION: Invoke the chain with the full context and query
            print("AI: ", end="", flush=True)
            
            # Streaming the response
            full_response = ""
            for chunk in rag_chain.stream({"question": user_input, "context": context}):
                content = ""
                
                # Check 1: Expected structured object with a '.content' attribute
                if hasattr(chunk, 'content'):
                    content = chunk.content
                # Check 2: Unexpected raw string (the issue you are seeing)
                elif isinstance(chunk, str): 
                    content = chunk
                
                if content:
                    print(content, end="", flush=True)
                    full_response += content

            end_time = time.time()
            print() # Newline after the full response is printed
            
            # 4. Source Citation
            print("\n" + "=" * 25)
            print(f"RAG Sources (Time: {end_time - start_time:.2f}s):")
            for i, doc in enumerate(retrieved_docs):
                # Show a snippet of the chunk that was used
                snippet = doc.page_content[:150].replace('\n', ' ') + "..."
                print(f"[{i+1}] Source: {doc.metadata.get('source', 'Unknown')} | Snippet: {snippet}")
            print("=" * 25)

        except Exception as e:
            # Handle the case where Ollama isn't running or Llama 3 isn't pulled
            if "ConnectionError" in str(e) or "Failed to connect" in str(e):
                print("\n[Connection Error: Please ensure the Ollama application is running (`ollama serve`).]")
                sys.exit(1)
            else:
                print(f"\nAn unexpected error occurred: {e}")
                
            continue

if __name__ == "__main__":
    try:
        # Before starting, check if Llama 3 is pulled and the server is available
        ollama.list() 
        
        # Build the RAG system
        retriever, rag_chain = setup_rag_pipeline()
        
        # Start the chat loop
        run_rag_chat(retriever, rag_chain)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not initialize RAG agent. Ensure Ollama is running and Llama 3 is pulled. Error: {e}")
        sys.exit(1)