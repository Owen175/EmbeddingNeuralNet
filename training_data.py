# save_wiki_100M.py
import re
import pickle
from datasets import load_dataset

# -----------------------------
# Parameters
# -----------------------------
MAX_TOKENS = 100_000_000      # Stop after this many tokens
OUTPUT_FILE = "wiki_tokens_100M.pkl"  # Pickle file to save tokens
LANGUAGE = "en"
DATASET_REVISION = "20231101.en"

# -----------------------------
# Load Wikipedia dataset (streaming)
# -----------------------------
print("Streaming Wikipedia dataset...")
dataset = load_dataset(
    "wikimedia/wikipedia",
    DATASET_REVISION,
    split="train",
    streaming=True  # stream to avoid downloading full dataset
)

# -----------------------------
# Tokenization and saving
# -----------------------------
token_count = 0
tokens = []

print("Processing articles and collecting tokens...")
for i, article in enumerate(dataset):
    text = article.get("text", "")
    if not text:
        continue
    
    # Simple tokenizer: split on words, lowercase
    article_tokens = re.findall(r"\b\w+\b", text.lower())
    
    # Check if adding all tokens exceeds MAX_TOKENS
    remaining = MAX_TOKENS - token_count
    if len(article_tokens) > remaining:
        tokens.extend(article_tokens[:remaining])
        token_count += remaining
        print(f"Reached {MAX_TOKENS} tokens. Stopping.")
        break
    else:
        tokens.extend(article_tokens)
        token_count += len(article_tokens)
    
    if i % 1000 == 0:
        print(f"Processed {i} articles, collected {token_count} tokens")

# -----------------------------
# Save tokens to disk (pickle)
# -----------------------------
print(f"Saving {len(tokens)} tokens to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(tokens, f)

print("Done! Tokens are ready for CBOW training.")