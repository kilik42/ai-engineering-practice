# Modern RAG example using current LangChain integration packages
#
# Install inside your active virtual environment:
# python -m pip install -U \
#   langchain-core \
#   langchain-text-splitters \
#   langchain-huggingface \
#   langchain-chroma \
#   # langchain-anthropic  # COMMERCIAL - intentionally not used \
#   sentence-transformers \
#   transformers \
#   torch \
#   chromadb \
#   python-dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_chroma import Chroma
# COMMERCIAL MODEL DISABLED:
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------

load_dotenv()

# COMMERCIAL API KEY SETUP DISABLED.
# This project is configured to use local/open-source models instead.
#
# api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
# if not api_key:
#     raise ValueError("No Anthropic API key found.")


# ------------------------------------------------------------
# Step 1: Sample document
# ------------------------------------------------------------

print("\nStep 1: Preparing our document")
print("-" * 50)

document = """
Artificial Intelligence (AI) is transforming the way we live and work. Machine learning,
a subset of AI, enables computers to learn from data without explicit programming.
Deep learning, a type of machine learning, uses neural networks inspired by the human brain.

Natural Language Processing (NLP) is a branch of AI that helps computers understand and
process human language. It's used in applications like translation, chatbots, and text analysis.

Computer Vision is another important field in AI. It enables machines to understand and
process visual information from the world, like images and videos. Applications include
facial recognition, autonomous vehicles, and medical image analysis.

Reinforcement Learning is a type of machine learning where agents learn by interacting
with an environment. They receive rewards for good actions and penalties for bad ones.
This is used in game playing, robotics, and autonomous systems.
"""

print("Document loaded. Length:", len(document), "characters")
print("\nPreview of the document:")
print(document[:200], "...\n")


# ------------------------------------------------------------
# Step 2: Chunking the document
# ------------------------------------------------------------

print("\nStep 2: Chunking the document")
print("-" * 50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ".", " "],
)

chunks = text_splitter.split_text(document)

print(f"Document has been split into {len(chunks)} chunks")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)
    print(f"\nChunk {i + 1} length: {len(chunk)} characters")
    print("-" * 50)


# ------------------------------------------------------------
# Step 3: Create embeddings
# ------------------------------------------------------------

print("\nStep 3: Creating embeddings")
print("-" * 50)

# This is the modern standalone Hugging Face integration.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings model loaded: sentence-transformers/all-MiniLM-L6-v2")
print(
    "This model converts text chunks into numerical vectors "
    "so semantically similar text can be found."
)


# ------------------------------------------------------------
# Step 4: Create a Chroma vector database
# ------------------------------------------------------------

print("\nStep 4: Creating a vector database and storing embeddings")
print("-" * 50)

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="ai_learning_demo",
)

print("Vector database created successfully.")
print(f"- Number of text chunks stored this run: {len(chunks)}")
print("- Persist directory: ./chroma_db")
print(f"- Embedding dimension: {len(embeddings.embed_query('test'))}")
print(f"- Vector store type: {type(vectorstore).__name__}")


# ------------------------------------------------------------
# Step 5: Query the vector database directly
# ------------------------------------------------------------

print("\nStep 5: Querying the vector database")
print("-" * 50)

query = "What is reinforcement learning?"

results = vectorstore.similarity_search(query, k=2)

print(f"Query: {query}")
print(f"Number of relevant chunks retrieved: {len(results)}")

for i, result in enumerate(results):
    print(f"\nRetrieved chunk {i + 1}:")
    print(result.page_content)
    print("-" * 50)


# ------------------------------------------------------------
# Step 6: Build the RAG pipeline explicitly
# ------------------------------------------------------------

print("\nStep 6: Setting up the RAG pipeline")
print("-" * 50)

# ------------------------------------------------------------
# OPEN-SOURCE / LOCAL GENERATION MODEL
# ------------------------------------------------------------
# Commercial Claude setup is intentionally disabled:
#
# llm = ChatAnthropic(
#     model="claude-sonnet-4-6",
#     api_key=api_key,
# )
#
# Instead, use an open-source Hugging Face model locally.
# FLAN-T5 is small enough for a learning/demo RAG pipeline.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_answer(prompt_text):
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a question-answering assistant using Retrieval-Augmented Generation (RAG).

Answer ONLY from the supplied context.

If the context does not contain enough information to answer the question,
say: "I don't have enough information in the supplied document to answer that."

Do not fill gaps using outside knowledge.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def ask_rag(question: str) -> dict:
    """
    Retrieve relevant document chunks, place them into the prompt,
    send the grounded prompt to the local open-source model, and return both the answer
    and the retrieved source documents.
    """

    # 1. RETRIEVE
    source_documents = retriever.invoke(question)

    # 2. AUGMENT
    context = "\n\n---\n\n".join(
        doc.page_content for doc in source_documents
    )

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # 3. GENERATE WITH LOCAL OPEN-SOURCE MODEL
    response = generate_answer(messages)

    return {
        "question": question,
        "answer": response,
        "source_documents": source_documents,
    }


print("RAG pipeline is ready.")


# ------------------------------------------------------------
# Example questions
# ------------------------------------------------------------

example_queries = [
    "What is reinforcement learning?",
    "Explain the concept of transfer learning.",
    "How does supervised learning differ from unsupervised learning?",
    "What are the main applications of computer vision?",
    "How is NLP used in real-world applications?",
]

for question in example_queries:
    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    try:
        response = ask_rag(question)

        print("\nANSWER:")
        print(response["answer"])

        print("\nRETRIEVED SOURCE CHUNKS:")
        for i, source_doc in enumerate(response["source_documents"], start=1):
            print(f"\nSource {i}:")
            print(source_doc.page_content)

    except Exception as e:
        print("\nERROR:")
        print(str(e))

print("\n" + "=" * 70)
print("RAG demo complete.")
