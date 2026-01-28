import pickle
import numpy as np
from helper_funcs import softmax
class Tester:
    def __init__(self, filename):
        with open(filename, "rb") as f:
            self.U, self.V = pickle.load(f)
        with open("word_to_info.pkl", "rb") as f:
            self.wti_dict = pickle.load(f)
        with open("idx_to_word_array.pkl", "rb") as f:
            self.itw_arr = pickle.load(f)
        
        self.vocabSize, self.hiddenLayerSize = self.U.shape

    def predict(self, contexts: list[str], n: int) -> list[str]:
        # List of words.
        indices = np.array([self.word_to_int(context) for context in contexts])
        h = np.mean(self.V[indices], axis=0)

        # Can softmax for probabilities instead
        return [self.int_to_word(x) for x in self.getHighestN(n, self.U @ h)]
        
    def getHighestN(self, n: int, vec: np.ndarray):
        # Get biggest n values by index
        top_n_indices_unsorted = np.argpartition(vec, -n)[-n:]

        # Sorts them
        return top_n_indices_unsorted[np.argsort(vec[top_n_indices_unsorted])[::-1]]

    def word_to_int(self, word: str) -> int:
        return self.wti_dict[word][1]
    
    def int_to_word(self, x: int) -> str:
        return self.itw_arr[x]