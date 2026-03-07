from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load semantic embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Take file paths from user
reference_file = input("Enter reference answer file path: ")
transcript_file = input("Enter transcript file path: ")

# Read files
with open(reference_file, "r", encoding="utf-8") as f:
    reference = f.read()

with open(transcript_file, "r", encoding="utf-8") as f:
    transcript = f.read()

# Generate embeddings
ref_embedding = model.encode(reference)
trans_embedding = model.encode(transcript)

# Compute similarity
similarity = cosine_similarity([ref_embedding], [trans_embedding])[0][0]

# Print only the similarity score
print(round(similarity * 100, 2), "%")