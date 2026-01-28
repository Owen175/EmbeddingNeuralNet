import numpy as np

# Returns the similarity between two vectors
def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.dot(x1, x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))

def get_negative_sample(n: int, arr: np.ndarray, exclude: int) -> np.ndarray:
    # rand = np.random.random() # 0-1
    # Now div + conquer the list
    # The array is the array of frequencies taken to the power of 3/4, normalised and made into a cdf. 
    output = np.zeros(n, dtype=int)
    for i in range(n):
        while 1:
            temp = np.searchsorted(arr, np.random.random(), side="right")
            if temp != exclude:
                output[i] = np.searchsorted(arr, np.random.random(), side="right")
                break

    return output



# I will represent one-hot vectors as an integer instead

# making V a col based embedding as in the paper
# Takes one-hot vectors in and gets the corresponding word vector
def onehots_to_embedded(V: np.ndarray, xs:np.ndarray) -> np.ndarray:
    # Imagine a is a vector with all 0s apart from idx x where there is a 1.
    # returning Va is equivalent to returning the column x
    return V[:,xs]

# Simple matrix-vector multiplication
def generate_score_vector(U: np.ndarray, x:np.ndarray) -> np.ndarray:
    return U.dot(x) 

def softmax(v: np.ndarray) -> None:
    # computes softmax of the array
    np.exp(v, out=v)
    v /= v.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# takes in a matrix who's rows are the embedded vectors and returns their average
def average_embedded(v: np.ndarray) -> np.ndarray:
    return v.mean(axis=0)


# Now a function which will get the word prediction from the word indices
def onehots_to_word(onehots: np.ndarray, V: np.ndarray, U: np.ndarray) -> np.ndarray:
    # onehots is a 1d array which contains each index for the onehot vectors
    embedded = onehots_to_embedded(V, onehots)
    avg = average_embedded(embedded)
    scorev = generate_score_vector(U, avg)
    softmax(scorev)
    return scorev

