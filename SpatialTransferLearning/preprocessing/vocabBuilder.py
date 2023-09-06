import pickle
from collections import Counter
import nltk
import yaml
import pandas as pd


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def add_vocab(args):
    
    """Build a simple vocabulary wrapper."""
    captions_data = pd.read_csv(args['image']['data_path'])
    captions = list(captions_data['caption'])
    counter = Counter()
    with open(args['vocab']['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)
    print(len(vocab))
    # Tokenize the collected captions
    for caption in captions:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= args['vocab']['threshold']]

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab



def main():
    args = yaml.safe_load(open("SpatialTransferLearning/preprocessing/params.yml"))
    updatedVocab = add_vocab(args)
    updated_vocab_path = args['vocab']['updated_vocab_path']
    with open(updated_vocab_path, 'wb') as f:
        pickle.dump(updatedVocab, f)
        
    print("Total vocabulary size: %d" %len(updatedVocab))
    print("Saved the vocabulary wrapper to '%s'" %updated_vocab_path)
    
if __name__ == '__main__':
    main()