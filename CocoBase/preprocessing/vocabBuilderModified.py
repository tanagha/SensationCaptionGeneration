import os
import pickle
from collections import Counter

import argparse
import nltk
from PIL import Image
from pycocotools.coco import COCO
import yaml



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

def build_vocab(json, threshold):
    
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids_coco = coco.anns.keys()
    
    # Create a dictionary to store captions for each image
    image_captions = {}
    ids = []
    c = 0
    # Iterate over the annotations to collect captions for each image
    for i, id in enumerate(ids_coco):
        image_id = coco.anns[id]['image_id']
        caption = str(coco.anns[id]['caption'])
        if image_id not in image_captions:
            image_captions[image_id] = {'image_id': image_id, 'id': id, 'caption': caption}
            ids.append(id)
            
             
        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))
    
    # Tokenize the collected captions
    for image_id, caption in image_captions.items():
        tokens = nltk.tokenize.word_tokenize(caption['caption'].lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab, image_captions, ids



def main():
    args = yaml.safe_load(open("C:/Users/anagh/Documents/myProject/Sensation Caption Generatioon/preprocessing/params.yml"))
    vocab, image_caption, ids = build_vocab(json=args['vocab']['caption_path'],threshold=args['vocab']['threshold'])
    vocab_path = args['vocab']['vocab_path']
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    imageCapDict_path = args['vocab']['imageCapDict_path']
    with open(imageCapDict_path, 'wb') as f:
        pickle.dump(image_caption, f)
    ids_path = args['vocab']['ids_path']
    with open(ids_path, 'wb') as f:
        pickle.dump(ids, f)
        
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)
    
if __name__ == '__main__':
    main()