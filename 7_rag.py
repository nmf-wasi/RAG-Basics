import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

MODEL = "gemini-3.5-flash-lite"
CUTOFF = 1.4
load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")  # loads the embedding model
client = chromadb.PersistentClient(
    path="chromadb"
)  # opens the database folder we already built. It does not create new data.
collection = client.get_collection(
    "notes"
)  # fetches an existing collection. It fails if the name is wrong,
# which is what we want, since get_or_create_collection would silently make an empty one.

question = "What's the capital fo France?"
query_embedding = model.encode(question).tolist()  # turns the question into a vector
results = collection.query(
    query_embeddings=[query_embedding], n_results=3
)  # returns the 3 nearest chunks

for dist, doc in zip(results["distances"][0], results["documents"][0]):
    print(f"{dist:.2f} | {doc[:60]!r}")

print(collection.count(), collection.metadata)


# Check which chunk contained our target word
# hit = collection.get(where_document={"$contains": "discrete categories"})
# print("chunks containing the phrase:", len(hit["ids"]))

# if hit["ids"]:
#     everything = collection.query(
#         query_embeddings=[query_embedding], n_results=collection.count()
#     )
#     for rank, (cid, d) in enumerate(
#         zip(everything["ids"][0], everything["distances"][0])
#     ):
#         if cid in hit["ids"]:
#             print(f"rank {rank} | dist {d:.2f}")


# puts the instruction, the chunks and the question into one block of text
# the whole “augmented” part of RAG. The model never sees your database, only this string.


gemini = genai.Client()


def ask(question):
    question_vector = model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_vector], n_results=3)
    context = "\n\n".join(results["documents"][0])

    top = results["distances"][0][0]
    if top > CUTOFF:
        # top is the distance of the closest chunk, since results come sorted small to large
        print(f"{top:.2f} | {question} -> Skipped (too far)")
        return
    print(results["distances"][0][0])

    # If it’s above CUTOFF, we print and return, which exits the function early. Gemini is never called.
    prompt = f"""Answer only from the text below.
        If the answer is not in the text, say "I don't know based on the notes." 

        TEXT:
        {context}

        QUESTION: {question}"""
    answer = gemini.models.generate_content(model=MODEL, contents=prompt).text
    print(f"{results['distances'][0][0]:.2f} | {question} -> {answer[:50]!r}")


ask("Which one predicts discrete categories?")
ask("What is the capital of France?")
ask("What is market-basket analysis?")
ask("Who won the 2022 World Cup?")
ask("How do I check if my model generalizes?")
