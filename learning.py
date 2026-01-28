# Loss function is J
# J= -log P(target word | context)

# https://arxiv.org/pdf/1411.2738

import numpy as np
from helper_funcs import softmax, sigmoid, get_negative_sample, cosine_similarity
from pickle import dump, load
# Structure - input layer -> hidden layer -> output layer


class Trainer:
    def __init__(self):
        self.vocabSize = 50428
        self.hiddenLayerSize = 150  # Same as the dimension of the embedding
        # not using an activation function

        self.V = np.random.randn(self.vocabSize, self.hiddenLayerSize) * 0.01
        self.U = np.random.randn(self.vocabSize, self.hiddenLayerSize) * 0.01
        # Weights matrices
        with open("Data/integer_filtered_100M.pkl", "rb") as f:  # Integer representations of each word w/o the rare words.
            self.tokens_array_ints = load(f)
        self.num_tokens = self.tokens_array_ints.size
        with open("Data/cum_distribution.plk", "rb") as f:
            self.cum_dist = load(f)
            
    def compute_score(self, x: np.ndarray) -> np.ndarray:
        h = self.V.T.dot(x)
        output = self.U.T.dot(h)
        softmax(output)
        return h, output  # Probability distribution
    
    def compute_score_list(self, l: int, r: int, pos: int) -> tuple[np.ndarray, np.ndarray, int]:
        # x is an array of inputs. for each input, you can calculate h, then avg and get the output from that. 
        
        numContext = r - l - 2 # r is one more than needed, and remove one for positive
        contexts = []
        for i in range(l, r):
            if i != pos:
                index = self.tokens_array_ints[i]
                contexts.append(index)
        h = np.mean(self.V[contexts], axis=0) 
        # Removed the softmax end as not needed for training
        return h, np.array(contexts), numContext

    def update_weights(self, bs: int, lr: float, idx: int, window_size: int, negative_samples: int):
        if idx + bs >= self.num_tokens: return  # the positive sample would end up as an error. Shouldn't be necessary but just in case. 

        storedUpdatesU = []
        storedUpdatesV = []
        for offset in range(bs):
            left = max(idx + offset - window_size, 0)
            right = min(idx + offset + window_size + 1, self.num_tokens - 1) # Add one to account for non-inclusive slicing
            negatives = get_negative_sample(negative_samples, self.cum_dist, idx+offset)
            h, contexts, contextNum = self.compute_score_list(left, right, idx + offset) # ----------------
            # negatives is the index of the words which are randomly selected as negatives samples. 
            
            # Setting up EH using old U so U can be updated. 
            posIdx = self.tokens_array_ints[idx + offset]
            smd = sigmoid(self.U[posIdx].dot(h))
            EH = (smd-1) * self.U[posIdx]
            
            for n in negatives:
                EH += sigmoid(self.U[n].dot(h)) * self.U[n]
            
            # Updating U
            storedUpdatesU.append((posIdx, -lr * (smd-1) * h))
            for n in negatives:
                storedUpdatesU.append((n, -lr * sigmoid(self.U[n].dot(h)) * h))

            # Updating V
            c = lr/contextNum  # Looping is faster than an inner product
            for i in contexts:
                storedUpdatesV.append((i, -c*EH))
        
        for (idx, change) in storedUpdatesU:
            self.U[idx] += change / bs
        for (idx, change) in storedUpdatesV:
            self.V[idx] += change / bs
        
    def train(self, epochs: int, batch_size: int, lr: float, min_lr: float, window_size: int, negative_samples: int, saves_per_epoch: int):
        # Training data of the form [([ipts], correct_hot_vector, [negatives])]
        # Need to split into blocks of the size of the batch size then feed to update_weights
        original_lr = lr
        lr_diff = original_lr - min_lr
        cycles_per_epoch = self.num_tokens // batch_size
        save_gap = cycles_per_epoch // saves_per_epoch
        total_cycles = cycles_per_epoch * epochs
        for epoch in range(epochs):
            print("Epoch ", epoch + 1)
            for i in range(cycles_per_epoch):
                lr = original_lr - (epoch * cycles_per_epoch + i)/(total_cycles) * lr_diff
                self.update_weights(batch_size, lr, i * batch_size, window_size, negative_samples)
                if i % 1000 == 0:
                    print("Epoch: ", epoch + 1, "  Cycle: ", i, " out of ", cycles_per_epoch)
                if i % save_gap == 0:
                    self.save()
                    self.evaluate(window_size)
    
    def save(self):
        with open("Data/UVsave.pkl", "wb") as f:
            dump((self.U, self.V), f)
        print("Saved matrices U and V")

    def evaluate(self, window: int) -> None:
        # take 1k points with context and:
        # - compare the cosine similarity of the embedding of the vector with contexts
        # - compare cosine similarity of the output with the ideal output - hot vector of correct output. 
        embedding_similarity = 0
        output_similarity = 0
        for _ in range(1000):
            targetIdx = np.random.randint(0, self.num_tokens)
            target = self.tokens_array_ints[targetIdx]
            target_embedding = self.V[target]

            contextRange = list(range(max(0, targetIdx - window), min(self.num_tokens, targetIdx + window + 1)))
            contextRange.remove(targetIdx)
            contextRange = [self.tokens_array_ints[idx] for idx in contextRange]
            contextEmbedding = np.mean(self.V[contextRange], axis=0)

            embedding_similarity += cosine_similarity(target_embedding, contextEmbedding)

            context_output = self.U @ contextEmbedding
            predicted_idx = np.argmax(context_output)

            if predicted_idx == target:
                output_similarity += 1
        print("Embedding cosine similarity: ", embedding_similarity / 1000, "\nOutput percentage accuracy: ", output_similarity/ 10, "%")