import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from scipy.ndimage.morphology import distance_transform_edt

from project import Project
from data.labels import get_pascal_labels


class VOC12Dataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.folder = Path(Project.data_dir / 'train.txt')
        self.folder_imgs = Path(Project.data_dir / 'JPEGImages')
        self.folder_masks = Path(Project.data_dir / 'SegmentationClass')
        self.img_paths = os.listdir(self.folder_imgs)
        self.mask_paths = os.listdir(self.folder_masks)
        self.transform = T.Compose([
            T.CenterCrop((256, 256)),
            T.ToTensor(),
        ])
        self.resize = T.CenterCrop((256, 256))

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, item):
        img = Image.open(
            self.folder_imgs + '/' + self.folder[item] + '.jpg').convert('RGB')
        img = self.transform(img)
        seg = Image.open(
            self.folder_masks + '/' + self.folder[item] + '.png').convert('RGB')
        seg = self.resize(seg)
        mask = np.asarray(seg).astype(int)
        for i, label in enumerate(get_pascal_labels()):
            if i == 0:
                s = torch.tensor(np.all(mask == label, axis=-1),
                                 dtype=torch.float).unsqueeze(dim=0)
                depth = distance_transform_edt(s.numpy())
            else:
                s = torch.cat((s, torch.tensor(np.all(mask == label, axis=-1),
                                               dtype=torch.float).unsqueeze(dim=0)), 0)
                depth += distance_transform_edt(s[-1].numpy())
        return img, s, torch.from_numpy(depth).squeeze(0)


