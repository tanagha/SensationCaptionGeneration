import torch
from utils import *
import time
from torch.nn.utils.rnn import pack_padded_sequence
import pickle
from torchvision import transforms
from preprocessing.vocabBuilder import *
from data_loader import get_loader


def tokens_to_text(tokens, vocab):
    return ' '.join([vocab.idx2word[idx] for idx in tokens])


def test(val_loader, encoder, decoder, vocab, device):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    references = list()  # references (true captions) 
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (imgs, caps, caplens) in enumerate(val_loader):

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)

        # Forward prop.
        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        
        references.extend(caps.tolist())
        referenecs_words = []
        # for ref in references:
        #     referenecs_words.append([vocab.idx2word[idx] for idx in ref if idx >1])
        
        for ref in references:
            referenecs_words.append(tokens_to_text(ref, vocab))
        
        
        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)
        hypotheses_words = []
        # for hyp in hypotheses:
        #     hypotheses_words.append([vocab.idx2word[idx] for idx in hyp])
        for hyp in hypotheses:
            hypotheses_words.append(tokens_to_text(hyp, vocab))
        assert len(referenecs_words) == len(hypotheses_words)

    

    return hypotheses_words



if __name__ == '__main__':
    # Define your encoder, decoder, vocab, and device
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("CocoBase/models/BEST_checkpoint_coco.pth.tar", map_location=torch.device('cpu'))
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    
    with open("CocoBase/data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    val_loader = get_loader('c:/Users/anagh/Documents/myProject/show_attend_and_tell_pytorch/data/val2014_resized/', 
                            'c:/Users/anagh/Documents/myProject/show_attend_and_tell_pytorch/data/annotations/captions_val2014.json',
                            vocab,
                            transform, 
                            10,
                            shuffle=True, 
                            num_workers=1)
    
    
    caption = test(val_loader, encoder, decoder, vocab, device)
