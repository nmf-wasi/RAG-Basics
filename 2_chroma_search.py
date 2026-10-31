import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-V2")

client = chromadb.PersistentClient(path="./chroma_db")
# saves the database to a folder called chroma_db on your disk.


sentences = [
    "I love myself",
    "I am in 20, about to turn 21",
    "Messi couldn't even touch the ball in final",
    "Jude kicked the ball so hard!",
    "Life is really hard thesedays",
    "I am writing codes in python",
    "Ronaldo scored a goal!",
]

embeddings = model.encode(sentences)

# collection = client.create_collection("sentences")

# no need of this collection cause for persistent client, we have get_or_create_collection

collection = client.get_or_create_collection("sentences")
# opens the collection if it exists and creates it if it doesn’t

collection.add(
    ids=[str(i) for i in range(len(sentences))],
    documents=sentences,
    embeddings=embeddings.tolist(),
)


query = "Who is struggling emotionally?"

query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3,
)

print(results["documents"][0])
print(results["distances"][0])

print(f"before : {collection.count()}")

collection.add(
    ids=[str(i) for i in range(len(sentences))],
    documents=sentences,
    embeddings=embeddings.tolist(),
)
# the ids are the same for those 7 sentences, since ids are same, they arent added as new sentences, therefore, before and after we will get 7 both times

# use of id : stops the same item from being stored twice


collection.upsert(
    ids=[str(i) for i in range(len(sentences))],
    documents=sentences,
    embeddings=embeddings.tolist(),
)

# the text and the vector are two separate things. Your new sentence also needs a new embedding, and that happens in model.encode(sentences), which already runs on every execution. If you change sentences but the encode line doesn’t run, the text and the vector will disagree.

# print(f"after add : {collection.count()}")

print(collection.get(ids=["0"])["documents"])
