from utils import *
import time
from torch.nn.utils.rnn import pack_padded_sequence


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, device, args):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        imgs = encoder(imgs)

        # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
        scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)

        targets = caps_sorted[:, 1:]
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(scores, targets)
        loss += args['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if args['grad_clip'] is not None:
            clip_gradient(decoder_optimizer, args['grad_clip'])
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args['grad_clip'])

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args['log_step'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    # torch.save(encoder.state_dict(), '/home/hpc/iwi5/iwi5149h/SensationCaptionGeneration/CocoBase/models/encoder_weights.pth')
    # torch.save(decoder.state_dict(), '/home/hpc/iwi5/iwi5149h/SensationCaptionGeneration/CocoBase/models/decoder_weights.pth')