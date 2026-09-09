# pip install langchain
# pip install langchain-text-splitter
# pip install langchain-community
# pip install langchain-anthropic
# pip install sentence-transformers
# pip install chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.chat_models import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv('API_KEY')

os.environ["Anthropic_API_KEY"] = api_key

# Step 1: Sample document
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
print(document[:200], "...\n")# Preview of the first 200 characters of the document


# Step 2: Chunking the document
print("\nStep 2: Chunking the document")
print("-" * 50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 3200, # Maximum number of characters in each chunk
    chunk_overlap =50, # Number of characters to overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", ".", " "] #this means the text will be split first by double newlines, then by single newlines, then by periods, and finally by spaces
)

chunks = text_splitter.split_text(document) # Split the document into smaller chunks based on the specified chunk size and overlap

print(f"Document has been split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks): # Iterate over each chunk and print its details
    print(f"\nChunk {i+1}:\n{chunk[:500]}...") # Preview of the first 500 characters of the chunk
    print(f"Chunk {i+1} length: {len(chunk)} characters") # Print the length of the chunk
    print(chunk) # Print the full content of the chunk
    print("-" * 50) # Separator between chunks


# Step 3: Create embeddings and store them in a vector database
# so llms only understand numbers not text, so we need to convert the chunnks into embeddings
# we will use a vector database to store these embeddings for efficient retrieval during question answering
# initialize embeddings

print("Step 3: Creating embeddings and storing them in a vector database")
print("-" * 50)

#huggingface embeddings can also be used as an alternative to OpenAIEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" # Specify the HuggingFace model to use for embeddings
    # we pick the model "all-MiniLM-L6-v2" because it provides a good balance between speed and accuracy for generating embeddings
) # Initialize the HuggingFace embeddings model

print("Embeddings model loaded.", embeddings.model_name)
print("this model will convert text chunks into numerical embeddings for efficient retrieval.")

# you could use open ai embeddings as an alternative to HuggingFaceEmbeddings
# embeddings = OpenAIEmbeddings() # Initialize the embeddings model

# Step 4: Create a vector database and store the embeddings
print("Step 4: Creating a vector database and storing the embeddings")
print("-" * 50)
vectorstore = Chroma.from_texts(
    texts= chunks, # Text chunks to be stored in the vector database
    embedding= embeddings, # Embeddings model to use for converting text chunks into numerical vectors
    persist_directory="./chroma_db" # Directory to persist the Chroma vector database
) # Create a Chroma vector database from the text chunks and embeddings
print("Vector database created and embeddings stored successfully.")
print("You can now use this vector database for efficient retrieval during question answering.")

print("RAG setup complete. You can now query the vector database for relevant chunks.")
print("Vector store is ready for querying.")
print(f"- number of text chunks stored: {len(chunks)}")
print(f"- persist directory: ./chroma_db")
print(f"- embedding dimension: {len(embeddings.embed_query('test'))}") # Print the dimension of the embeddings generated by the model
print(f"- vector store type: {type(vectorstore)}") # Print the type of the vector store to confirm it is a Chroma instance
print(f"- vector store location: {vectorstore.persist_directory}") # Print the location where the vector store is persisted
print("You can now use the vector store to perform similarity searches and retrieve relevant text chunks.")


#step 5: Query the vector database for relevant chunks
print("Step 5: Querying the vector database for relevant chunks")
print("-" * 50)
query = "Enter your query here" # Replace with your actual query
results = vectorstore.similarity_search(query) # Perform a similarity search in the vector database
print(f"Query: {query}") # Print the query being used for the similarity search
print(f"Number of relevant chunks retrieved: {len(results)}") # Print the number of relevant chunks retrieved from the vector database
for i, result in enumerate(results): # Iterate over the retrieved relevant chunks and print their content
    print(f"Chunk {i+1}: {result.page_content}") # Print the content of each retrieved relevant chunk


query ="what is reinforcement learning" # Replace with your actual query
results = vectorstore.similarity_search(query, k=2) # Perform a similarity search in the vector database 
#k specifies the number of top relevant chunks to retrieve from the vector database
print(f"Query: {query}") # Print the query being used for the similarity search
print(f"Number of relevant chunks retrieved: {len(results)}") # Print the number of relevant chunks retrieved from the vector database
for i, result in enumerate(results): # Iterate over the retrieved relevant chunks and print their content
    print(f"Chunk {i+1}: {result.page_content}") # Print the content of each retrieved relevant chunk
    print("-" * 50) # Print a separator line between retrieved chunks
    print() # Print an empty line for better readability between different querie

# You can repeat the above steps with different queries to retrieve relevant chunks from the vector database as needed

# step 6 set up rag pipeline
print("Step 6: Setting up the RAG pipeline")
print("-" * 50)

print("RAG pipeline setup is in progress...") # Print a message indicating the setup of the RAG pipeline
print("using anthropic model with the provided api key")

print("RAG pipeline is now ready to handle queries.") # Print a message indicating the RAG pipeline is ready


# paid model setup for Anthropic's Claude model
llm = ChatAnthropic(
    model = "claude-3-5-sonnet-2024060",
    temperature=0.7,
    api_key="your_anthropic_api_key") # Replace with your actual API key for the Anthropic model


# this is an example of setting up the RAG pipeline with the open ai model
# OpenAI:
# if "OPENAI_API_KEY" in os.environ:
#     llm = OpenAI(
#         model_name="gpt-4",
#         temperature=0.5
#     )
#

# HuggingFace:
# if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
llm = HuggingFaceHub(
        repo_id="google/flan-t5-small", # The HuggingFace model repository ID for the Flan-T5 small model
        model_kwargs={"temperature": 0.5, "max_length": 512} # The model parameters for the HuggingFace model
    )



# create the RAG pipeline using the selected LLM and the vectorstore
rag_pipeline = RetrievalQA(
    llm=llm,
    chain_type="stuff", # Specify the type of chain to use in the RAG pipeline
    retriever=vectorstore.as_retriever(), # Use the vectorstore as the retriever for the RAG pipeline
    return_source_documents=True # Indicate whether to return the source documents along with the retrieved answers
)


# example questions to ask the RAG pipeline
example_queries = [
    "What is reinforcement learning?",
    "Explain the concept of transfer learning.",
    "How does supervised learning differ from unsupervised learning?",
    "What is reinforcement learning and how does it work?",
    "What are the main applications of computer vision?",
    "How is NLP used in real-world applications?"
]

# iterate over the example queries and ask the RAG pipeline for answers
for query in example_queries:
    print("\nQuestion:", query)
    try: # Attempt to invoke the RAG pipeline with the current query
        response = rag_pipeline.invoke(query)# we use invoke instead of run for the RAG pipeline because it is the recommended method for invoking the pipeline. Run is deprecated.
    except Exception as e: # Handle any exceptions that occur during the invocation of the RAG pipeline
        response = f"An error occurred: {str(e)}"
    print(f"Query: {query}") # Print the current query being asked to the RAG pipeline
    print(f"Response: {response}") # Print the response returned by the RAG pipeline
    print("-" * 50)

