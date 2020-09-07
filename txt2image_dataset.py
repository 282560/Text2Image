import io
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PIL import Image
import torch

class Text2ImageDataset(Dataset):

    def __init__(self, datasetFile, transform=None, split=0):
        self.datasetFile = datasetFile
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        # split == 0   -> 'train' | dla kwiatow - 29 390
        # split == 1   -> 'valid' | dla kwiatow -  5 780
        # split != 0&1 -> 'test'  | dla kwiatow -  5 775
        #                           dla kwiatow - 40 945 na 8 189 obrazow (40945 : 8189 = 5)
        self.h5py2int = lambda x: int(np.array(x))

    def __len__(self):
        f = h5py.File(self.datasetFile, 'r')
        self.dataset_keys = [str(k) for k in f[self.split].keys()]

        length = len(f[self.split])
        f.close()

        return length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

        #self.CheckingImagesNames(max_idx=self.__len__())
        #exit()

        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]

        right_image = np.array(example['img']).tobytes()
        right_embed = np.array(example['embeddings'], dtype=float)
        wrong_image = np.array(self.find_wrong_image(example['class'])).tobytes()
        inter_embed = np.array(self.find_inter_embed())

        if example_name == 'image_00003_0':
            exit()

        print('Name:', example_name, '\n')
        print('right_embed:')
        print(right_embed)

        #print('\nvector length:', len(right_embed))

        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))

        a = example['txt'][()]
        special = u"\ufffd\ufffd"
        a = a.replace(special,' ')
        txt = np.array(a).astype(str)
        txt = str(txt)

        print('idx:', idx)
        print('\ntxt:', txt)
        '''
        exit()
        
        if self.split == 'test':
            name = txt.replace("/", "").replace("\n", "").replace(" ", "_")[:100]
            right_image.save('results_demo/original_images/{0}.jpg'.format(name))

        right_image = self.validate_image(right_image)
        wrong_image = self.validate_image(wrong_image)
        sample = {
                'right_images': torch.FloatTensor(right_image),
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_images': torch.FloatTensor(wrong_image),
                'inter_embed': torch.FloatTensor(inter_embed),
                'txt': txt }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] = sample['wrong_images'].sub_(127.5).div_(127.5)
        return sample'''

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']


    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

    def CheckingImagesNames(self, max_idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

        for i in range(max_idx):
            example_name = self.dataset_keys[i]
            print(example_name)