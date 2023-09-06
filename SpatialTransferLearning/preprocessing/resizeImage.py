import os
from PIL import Image
import yaml
import pandas as pd
import pickle

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.LANCZOS)
    return image


def main():
    args = yaml.safe_load(open("SpatialTransferLearning/preprocessing/params.yml"))
    captions = pd.read_csv(args['image']['data_path'])
    
    folder = args['image']['folder']
    resized_folder = args['image']['resized_folder']
    
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
        
    print('Start resizing images.')
    
    image_files = list(captions['image_path'])
    num_images = len(image_files)
    
    list_duplicate = []
    for i, image_file in enumerate(image_files):
        with open(os.path.join(folder + image_file), 'r+b') as f:
            with Image.open(f) as image:
                image = resize_image(image)
                save_name = resized_folder+ '/' + os.path.splitext(os.path.basename(image_file))[0] + '.png'
                if os.path.exists(save_name):
                    print('Resized image already exists: %s' % save_name)
                    list_duplicate.append(save_name)
                    pass  # Stop the loop
                    
                image.save(save_name)
        if i % 100 == 0:
            print('Resized images: %d/%d' %(i, num_images))
    
    if len(list_duplicate)>0:
        print("There were some duplicates, please check")
        with open('SpatialTransferLearning/data/dupliacteList.pkl', 'wb') as f:
            pickle.dump(list_duplicate, f)
    else: 
        print("no duplicates found")
            
if __name__ == '__main__':
    main()