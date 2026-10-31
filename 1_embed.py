from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")

sentences=[
    "I love myself",
    "I am in 20, about to turn 21",
    "Messi couldn't even touch the ball in final",
    "Jude kicked the ball so hard!",
    "Life is really hard thesedays",
    "I am writing codes in python",
    "Ronaldo scored a goal!",
]

embeddings=model.encode(sentences)

print(embeddings.shape)
print(embeddings[0][:5])

# Before the model runs, the sentences are turned into tokens, 
# and shorter ones are padded so they all have the same number of tokens 
# and can be processed together as a batch. 
# But the model then squashes all those tokens into one vector per sentence. 
# That squashing step is why a 4-word sentence and a 12-word sentence both come out as 384 numbers.

# how close are two vectors”. The standard measure is cosine similarity. 
# It measures the angle between two vectors:

# close to 1 means pointing the same direction, so similar meaning
# close to 0 means unrelated
# negative means opposite (rare in practice)

import numpy as np
def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# print(cosine_similarity(embeddings[2], embeddings[3])) # football vs football
# print(cosine_similarity(embeddings[2], embeddings[-1])) # football vs football
# print(cosine_similarity(embeddings[0], embeddings[4])) # love vs life
# these models rarely give high scores for sentences that use different words and describe different events
# embeddings is only computed when model.encode(sentences) runs. 
# If you edit the sentences list but the embeddings = model.encode(sentences) line runs before your edit, or you forget it, 
# embeddings[-1] points at a stale vector, not your new sentence. After you change the list, always re-encode. 


query="Who is struggling emotionally?"
query_embedding=model.encode(query)
# turns the question into a vector. You use the same model as for the sentences.

scores=[]

for i, sentence in enumerate(sentences):
    # gives you the position i and the sentence together, so embeddings[i] is the vector for that sentence
    score=cosine_similarity(query_embedding, embeddings[i])
    scores.append((score, sentence))

scores.sort(reverse=True)

for score, sentence in scores[:3]:
    print(round(float(score),3), sentence)

