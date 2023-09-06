import nltk
import torch
from PIL import Image
import pandas as pd

class TrafficLightDataLoader(torch.utils.data.Dataset):
    def __init__(self, image_path_main, caption_path, vocab, transform=None):
        self.vocab = vocab
        self.transform = transform
        self.image_path_main = image_path_main
        self.caption_path=caption_path
        img_cap_data = pd.read_csv(self.caption_path)
        self.img_caption_dict = []
        for _, row in img_cap_data.iterrows():
            image_name = row['image_path']
            caption = row['caption']
            self.img_caption_dict.append({'image': image_name, 'caption': caption})
                  
    def __getitem__(self, index):
        vocab = self.vocab
        img_id = self.img_caption_dict[index]
        image_path = self.image_path_main + img_id['image']
        caption = img_id['caption'] 
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.img_caption_dict)

def collate_fn(data):
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

# def get_loader(image_path_main, caption_path, vocab, transform, batch_size, shuffle, num_workers):
#     traffic = TrafficLightDataLoader(image_path_main=image_path_main, caption_path = caption_path, vocab=vocab, transform=transform)
    
#     # Data loader for Traffic dataset
#     # This will return (images, captions, lengths) for every iteration.
#     # images: tensor of shape (batch_size, 3, 224, 224).
#     # captions: tensor of shape (batch_size, padded_length).
#     # lengths: list indicating valid length for each caption. length is (batch_size).
#     data_loader = torch.utils.data.DataLoader(dataset=traffic,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               num_workers=num_workers,
#                                               collate_fn=collate_fn)
    
    
#     return data_loader


def get_loader(image_path_main, caption_path, vocab, transform, batch_size, shuffle, num_workers, train_ratio=0.8):
    traffic = TrafficLightDataLoader(image_path_main=image_path_main, caption_path=caption_path, vocab=vocab, transform=transform)
    
    dataset_size = len(traffic)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(traffic, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               collate_fn=collate_fn)
    
    test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,  # No need to shuffle test data
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    
    return train_loader, test_loader