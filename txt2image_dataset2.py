from corpus_loader import CorpusLoader
from torch.utils.data import Dataset
from PIL import Image

import tensorlayer as tl
import numpy as np

import easydict
import random
import torch
import math
import os

class Text2ImageDataset2(Dataset):

    def __init__(self, datasetFile, imagesDir, textDir, split, arrangement, sampling):
        self.datasetFile = datasetFile
        self.imagesDir = imagesDir
        self.textDir = textDir
        self.split = split
        self.arrangement = easydict.EasyDict(arrangement)
        self.sampling = easydict.EasyDict(sampling)

        self.images_classes = {}
        self.assign_classes()

        cl = CorpusLoader(datasetFile=datasetFile, textDir=textDir)
        self.vectorizer = cl.TrainVocab()

    def __len__(self):
        if self.split == 'train':
            return self.arrangement.train * self.sampling.train
        elif self.split == 'valid':
            return self.arrangement.valid * self.sampling.valid
        elif self.split == 'test':
            return self.arrangement.test * self.sampling.test
        else:
            return -1

    def __getitem__(self, idx):
        image_index = math.floor(idx / self.sampling.train) + 1  # Domyslnie dla 'train'
        caption_index = idx % self.sampling.train                # Domyslnie dla 'train'
        if self.split == 'valid':
            image_index = self.arrangement.train + image_index
            caption_index = idx % self.sampling.valid
        elif self.split == 'test':
            image_index = self.arrangement.train + self.arrangement.valid + image_index
            caption_index = idx % self.sampling.test

        img_dir = os.path.join(self.datasetFile, self.imagesDir)
        name = self.complete_name(idx=image_index, ext='jpeg')

        right_image = Image.open(os.path.join(img_dir, name)).resize((64, 64))
        right_image = self.validate_image(right_image)

        wrong_image_name = self.find_wrong_image_name(name, self.images_classes, self.split)
        wrong_image = Image.open(os.path.join(img_dir, wrong_image_name)).resize((64, 64))
        wrong_image = self.validate_image(wrong_image)

        caption_name = self.complete_name(idx=image_index, ext='txt')
        caption_dir = os.path.join(self.datasetFile, self.textDir)
        caption_sub_dir = os.path.join(caption_dir, self.images_classes[name], caption_name)

        txt = ''
        with open(caption_sub_dir, "r") as file:
            for i, line in enumerate(file):
                if i == caption_index:
                    txt = line

        captions = []
        captions.append(txt)

        vector = self.vectorizer.transform(captions)
        raw_vector = vector.toarray()
        right_embed = np.array(raw_vector[0])

        sample = {
            'right_images': torch.FloatTensor(right_image),
            'right_embed': torch.FloatTensor(right_embed),
            'wrong_images': torch.FloatTensor(wrong_image),
            'txt': txt
        }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] = sample['wrong_images'].sub_(127.5).div_(127.5)

        return sample

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

    def complete_name(self, idx, ext):
        num = idx
        positions = []
        while num != 0:
            positions.append(num % 10)
            num = num // 10
        complete = ((5 - len(positions)) * '0')
        for i in reversed(positions):
            complete = complete + str(i)
        return 'image_' + complete + '.' + ext

    def find_wrong_image_name(self, img_name, images_classes, split):
        img_class = images_classes[img_name]
        while(True):
            rnd = random.randint(1, self.__len__() - 1)
            image_index = -1
            if split == 'train':
                image_index = math.floor(rnd / self.sampling.train) + 1
            elif split == 'valid':
                image_index = math.floor(rnd / self.sampling.valid) + 1
            elif split == 'test':
                image_index = math.floor(rnd / self.sampling.test) + 1
            wrong_img_name = self.complete_name(idx=image_index, ext='jpeg')
            wrong_img_class = images_classes[wrong_img_name]
            if (img_name != wrong_img_name) and (img_class != wrong_img_class):
                return wrong_img_name

    def load_folder_list(self, path=''):
        return [os.path.join(path,o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]

    def assign_classes(self):
        caption_dir = os.path.join(self.datasetFile, self.textDir)
        caption_sub_dir = self.load_folder_list(caption_dir)

        for sub_dir in caption_sub_dir:
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')

            for file in files:
                pre, ext = os.path.splitext(file)
                self.images_classes[pre + '.jpeg'] = os.path.basename(os.path.normpath(sub_dir))

        self.images_classes = dict(sorted(self.images_classes.items())) # Posortowane po kluczu