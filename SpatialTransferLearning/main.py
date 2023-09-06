import os, sys
from datetime import datetime
import pickle
from preprocessing.vocabBuilder import *
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from model import EncoderCNN, AttnDecoderRNN

from data_loader import get_loader

from utils import *
import yaml
from train import *
from validation import *


sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = yaml.safe_load(open("SpatialTransferLearning/config.yml"))

def main(args):
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch

    # Format the current time as a string (optional)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = args['model_path'] + timestamp_str
    # Create the folder
    os.makedirs(checkpoint_dir)
    
    # Load vocabulary wrapper
    with open(args['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)
        
    if args['transfer_learning']:
        checkpoint = torch.load(args['checkpoint'], map_location=torch.device('cpu'))
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        fine_tune_encoder = args['fine_tune_encoder']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                lr=args['encoder_lr'])
    else:               
        if args['checkpoint'] is None:
            decoder = AttnDecoderRNN(attention_dim=args['attention_dim'],
                                    embed_dim=args['embed_dim'],
                                    decoder_dim=args['decoder_dim'],
                                    vocab_size=len(vocab),
                                    dropout=args['dropout'])
            decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),lr=args['decoder_lr'])
            encoder = EncoderCNN()
            encoder.fine_tune(args['fine_tune_encoder'])
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                lr=args['encoder_lr']) if args['fine_tune_encoder'] else None
        else:
            checkpoint = torch.load(args['checkpoint'])
            start_epoch = checkpoint['epoch'] + 1
            epochs_since_improvement = checkpoint['epochs_since_improvement']
            best_bleu4 = checkpoint['bleu-4']
            decoder = checkpoint['decoder']
            decoder_optimizer = checkpoint['decoder_optimizer']
            encoder = checkpoint['encoder']
            encoder_optimizer = checkpoint['encoder_optimizer']
            fine_tune_encoder = args['fine_tune_encoder']
            if fine_tune_encoder is True and encoder_optimizer is None:
                encoder.fine_tune(fine_tune_encoder)
                encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                    lr=args['encoder_lr'])
    
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Build data loader
    train_loader, val_loader = get_loader(args['image_dir'], args['caption_path'], vocab,
                              transform, args['batch_size'],
                              shuffle=True, num_workers=args['num_workers'])
    
    for epoch in range(args['start_epoch'], args['epochs']):
        if args['epochs_since_improvement'] == 20:
            break
        if args['epochs_since_improvement'] > 0 and args['epochs_since_improvement'] % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args['fine_tune_encoder']:
                adjust_learning_rate(encoder_optimizer, 0.8)
        
        
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              device=device,
              args=args)
        
        
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                vocab = vocab,
                                device=device,
                                args=args)

        is_best = recent_bleu4 > args['best_bleu4']
        best_bleu4 = max(recent_bleu4, args['best_bleu4'])
        if not is_best:
            args['epochs_since_improvement'] +=1
            print ("\nEpoch since last improvement: %d\n" %(args['epochs_since_improvement'],))
        else:
            args['epochs_since_improvement'] = 0
        
        save_checkpoint(checkpoint_dir, args['data_name'], args, epoch, args['epochs_since_improvement'], encoder, decoder, encoder_optimizer, decoder_optimizer,
                        recent_bleu4, is_best)



if __name__ == '__main__':
    main(args)
