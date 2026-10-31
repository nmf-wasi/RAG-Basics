import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()  # in memory, starts empty every run
collection = client.create_collection("test")

# store the original sentences
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

collection.add(
    ids=[str(i) for i in range(len(sentences))],
    documents=sentences,
    embeddings=embeddings.tolist(),
)


print("Step 1 count:", collection.count())
print("Step 1 id 0 :", collection.get(ids=["0"])["documents"])

# try to change id 0 using add 
new_text = ["I love football"]

collection.add(
    ids=["0"],
    documents=new_text,
    embeddings=model.encode(new_text).tolist(),
)
print("Step 2 count:", collection.count())
print("Step 2 id 0 :", collection.get(ids=["0"])["documents"])

# change id 0 using upsert
collection.upsert(
    ids=["0"],
    documents=new_text,
    embeddings=model.encode(new_text).tolist(),
)
print("Step 3 count:", collection.count())
print("Step 3 id 0 :", collection.get(ids=["0"])["documents"])
