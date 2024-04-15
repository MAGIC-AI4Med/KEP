from torch.utils.data import Dataset
from PIL import Image, ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class CLIPImageCaptioningDataset(Dataset):
    def __init__(self, df, preprocessing):
        self.images = df["image"].tolist()
        self.caption = df["caption"].tolist()
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = self.preprocessing(Image.open(self.images[idx]).convert('RGB'))  # preprocess from clip.load
        caption = self.caption[idx]
        return images, caption


class CLIPCaptioningDataset(Dataset):
    def __init__(self, captions):
        self.caption = captions

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        caption = self.caption[idx]
        return caption


class CLIPImageDataset(Dataset):
    def __init__(self, img_dir_list, preprocessing):
        self.images = img_dir_list
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(self.preprocessing,dict) and 'imgprocessor' in self.preprocessing:
            img = Image.open(self.images[idx]).convert('RGB')
            try:
                images = self.preprocessing['imgprocessor'](images=img, return_tensors="pt")
            except:
                images = self.preprocessing['imgprocessor'](img)
        else:
            images = self.preprocessing(Image.open(self.images[idx]).convert('RGB'))  # preprocess from clip.load
        return images

        
class CLIPImageLabelDataset(Dataset):
    def __init__(self, df, preprocessing):
        self.images = df["image"].tolist()
        self.label = df["label"].tolist()
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.preprocessing(Image.open(self.images[idx]).convert('RGB'))  # preprocess from clip.load
        label = self.label[idx]
        return images, label