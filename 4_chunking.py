def chunk_text(text, chunk_size=200, overlap=40):

    chunks = []
    start = 0  # start = 0 is where the current chunk begins
    while start < len(text):
        end = start + chunk_size
        chunks.append(
            text[start:end]
        )  # text[start:end] cuts out that piece of the text
        start = (
            end - overlap
        )  # moves the start back by 40 characters, so the next chunk begins inside the end of the previous one.
        # Without - overlap, chunks would not share any text

    return chunks


text = (
    "Machine learning is a field of AI where computers learn patterns from data. "
    "Supervised learning uses labeled examples, such as emails marked spam or not spam. "
    "Unsupervised learning finds structure in data without labels, like grouping customers. "
    "Neural networks are models made of layers of connected units that learn features. "
    "Overfitting happens when a model memorizes training data and fails on new data."
)

chunks = chunk_text(text)
# print(chunks)

# for i, c in enumerate(chunks):
#     print(i, len(c), repr(c))
#     print()

# OUTPUT : [breaks words in the middle which is problematic]
# 0 200 'Machine learning is a field of AI where computers learn patterns from data. Supervised learning uses labeled examples, such as emails marked spam or not spam. Unsupervised learning finds structure in '
# 1 200 'nsupervised learning finds structure in data without labels, like grouping customers. Neural networksare models made of layers of connected units that learn features. Overfitting happens when a model'
# 2 87 'atures. Overfitting happens when a model memorizes training data and fails on new data.'

import re


def chunk_by_sentences(text, max_chars=200):
    sentences = re.split(
        r"(?<=[.!?])\s+", text
    )  # cuts the text after every ., ! or ? that is followed by whitespace.
    # The punctuation stays attached to its sentence.

    chunks = []
    current = ""  # current is the chunk being built
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            # would adding this sentence (plus one space) push the chunk over the limit?
            # If yes, save current as a finished chunk and start a new one with this sentence
            chunks.append(current)
            current = sentence
        else:
            current = (current + " " + sentence) if current else sentence

    if current:
        chunks.append(current)
    return chunks


chunks = chunk_by_sentences(text)
for i, c in enumerate(chunks):
    print(i, len(c), repr(c))
