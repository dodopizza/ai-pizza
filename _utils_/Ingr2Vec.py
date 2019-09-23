import numpy as np
class WordVectors:
    
    def __init__(self, vocabulary, embedding_matrix, lemmatizer):
        self.vocab = vocabulary
        self.W = embedding_matrix
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.lemmatizer = lemmatizer
        
    def lemmatize(self, word):
        return self.lemmatizer.lemmatize(word)
    
    def word_vector(self, word):
        """ 
        Takes word and returns its word vector.
        """
        num = self.vocab[word]
        word_vector = self.W[num, :]

        return word_vector
    
    def nearest_words(self, word, top_n= 10):
        """ 
        Takes word from the vocabulary and returns its top_n
        nearest neighbors in terms of cosine similarity.
        """
        neighbors = list()
        w_vector = self.word_vector(word)
        
        cosine = np.zeros((1, self.W.shape[0]))
        
        for i in range(self.W.shape[0]):
            emb_vec = self.W[i, :]
            cosine[0, i] = (w_vector.T @ emb_vec) / (np.linalg.norm(w_vector) * np.linalg.norm(emb_vec))
        
        cosine[0, self.vocab[word]] = float('-inf')
        
        tops = np.argsort(cosine[0, :])
        
        for i in reversed(tops[-top_n:]):
            neighbors.append((self.inv_vocab[i], np.round(cosine[0, i], 3)))

        return neighbors

class IngrVectors(WordVectors):
    def ingr_vector(self, ingr):
        """ 
        Takes ingredient and returns its word vector.
        """
        word_vector = np.zeros(self.W.shape[1])
        for word in ingr.split('_'):
            prepr_word = self.lemmatizer.lemmatize(word)
            num = self.vocab[prepr_word]
            word_vector += self.W[num, :]

        return word_vector
    
def default_model(filepath=r'_utils_/Ingr2Vec.npy'):
    return np.load(filepath).item()