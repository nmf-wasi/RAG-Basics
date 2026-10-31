from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection("notes")

# read and chunk
text = Path("notes.txt").read_text(
    encoding="utf-8"
)  # reads the whole file into one string.
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_text(text)

print(f"chunks : {len(chunks)}")

# embed and store
embeddings = model.encode(chunks).tolist()
collection.upsert(  # upsert -> updates instead of skipping even when it reruns
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
    metadatas=[
        {"source": "notes.txt", "chunk": i} for i in range(len(chunks))
    ],  # stores extra info with each chunk (which file, which chunk number)
    # this is how you tell the user where an answer came from.
)

print("stored : ", collection.count())

# ask questions
questions = [
    "What is the main difference between traditional programming and machine learning?",
    "What do the seeds, nutrients, gardener, and plants represent in the machine learning gardening analogy?",
    "What are the three core components present in every machine learning algorithm?",
    "What does the equation y = f(x) represent in the ML prediction framework?",
    "What are the separate roles of the training, validation, and test datasets?",
    "Name two specific scenarios where machine learning is preferred over explicit human programming.",
    "What does the probability expression P(chips | beer) = 0.7 mean in market-basket analysis?",
    "What specific role does Statistics play in Machine Learning compared to Computer Science?",
    "What are the first three steps in the standard Machine Learning project workflow?",
    "Which ML task type would you use if you want to detect anomalies versus predicting a continuous quantity?",
    "What are the last two steps in the ML workflow?",
]

for q in questions:
    results = collection.query(
        query_embeddings=[model.encode(q).tolist()],
        n_results=3,
    )
    print("\nQ:", q)
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  chunk {meta['chunk']}  dist={dist:.2f}  {doc[:60]!r}")

# if you later change chunk_size, the new run may produce fewer chunks, and the old extra chunks stay in the database from the previous run.
# Delete the chroma_db folder before each experiment so you start clean.
