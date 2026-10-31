from langchain_text_splitters import RecursiveCharacterTextSplitter

text = (
    "Machine learning is a field of AI where computers learn patterns from data. "
    "Supervised learning uses labeled examples, such as emails marked spam or not spam. "
    "Unsupervised learning finds structure in data without labels, like grouping customers. "
    "Neural networks are models made of layers of connected units that learn features. "
    "Overfitting happens when a model memorizes training data and fails on new data."
)

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)

chunks = splitter.split_text(text)

for i, c in enumerate(chunks):
    print(i, len(c), repr(c))


long_text = "This is one very long sentence with no full stops anywhere in it " * 6
print(splitter.split_text(long_text))