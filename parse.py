import pickle
import numpy as np
from collections import Counter
# import pickle
# with open("wiki_tokens_100M.pkl", "rb") as f:
#     tokens = pickle.load(f)  # binary pickle read 

# token_freq = Counter(tokens)

with open("idx_to_word.pkl", "rb") as f:
    tokens = pickle.load(f)
print(tokens)
a = []

for key, val in tokens.items():
    a.append(val)

with open("idx_to_word_array.pkl", "wb") as f:
    pickle.dump(np.array(a), f)