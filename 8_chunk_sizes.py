from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print(len(model.encode("hi")))
print(len(model.encode("machine Learning")))
print(model.max_seq_length)

text="machine learning"*500
print(len(model.tokenizer(text)["input_ids"]))
print(len(model.encode(text)))