import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
CUTOFF = 1.4

model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="chromadb")
collection = db.get_collection("notes")
gemini = genai.Client()


def ask(question):
    question_vector = model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_vector], n_results=3)
    top = results["distances"][0][0]
    if top > CUTOFF:
        print(f"{top:0.2f} | {question} -> SKIPPED (too far)")
        return

    context = "\n\n".join(results["documents"][0])
    prompt = f"""Answer only from the text below. If the answer is not in the text, say "I don't know from the notes"
TEXT:
{context}
    
QUESTION: {question}
    """

    answer = gemini.models.generate_content(model=MODEL, contents=prompt).text
    print(f"{top:0.2f} | {question} -> {answer[:50]!r}")


ask("Which one predicts discrete categories?")
ask("What is the capital of France?")
ask("What is market-basket analysis?")
ask("Who won the 2022 World Cup?")
ask("How do I check if my model generalizes?")
