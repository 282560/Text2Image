from PIL import Image

import numpy as np

import h5py
import io
import os

class HDF2files():

    def __init__(self, path, main_dir_name, img_dir_name, txt_dir_name):
        self.path = path
        self.main_dir_name = main_dir_name
        self.img_dir_name = img_dir_name
        self.txt_dir_name = txt_dir_name

    def complete_name(self, full_name, idx, ext=''):
        num = idx
        positions = []
        while num != 0:
            positions.append(num % 10)
            num = num // 10
        fullfill = ((5 - len(positions)) * '0')
        for i in reversed(positions):
            fullfill = fullfill + str(i)
        if ext == '':
            return full_name + fullfill
        else:
            return full_name + fullfill + '.' + ext

    def convert(self):
        datasets = ['train', 'test', 'valid']

        hdf1 = h5py.File(self.path, mode='r')
        all_samples = sum([len(hdf1[dataset]) for dataset in datasets])

        iterator = 1
        all_data = {}
        all_classes = []

        print('Reading HDF5...')
        with h5py.File(self.path, mode='r') as hdf2:
            for dataset in datasets:
                dataset_keys = [str(k) for k in hdf2[dataset].keys()]
                length = len(hdf2[dataset])
                for idx in range(0, length):
                    dataset_name = dataset_keys[idx]
                    example = hdf2[dataset][dataset_name]

                    example_name = example['name'][()]
                    example_class = example['class'][()]

                    a = example['txt'][()]
                    special = u"\ufffd\ufffd"
                    a = a.replace(special, ' ')
                    txt = np.array(a).astype(str)
                    example_txt = str(txt)

                    example_img = np.array(example['img']).tobytes()
                    example_img = Image.open(io.BytesIO(example_img))

                    if not example_class in all_classes:
                        all_classes.append(example_class)

                    example_class_txt_img = (example_class, example_txt, example_img)
                    if not example_name in all_data:
                        all_data[example_name] = [example_class_txt_img]
                    else:
                        all_data[example_name].append(example_class_txt_img)

                    iterator += 1
        print('HDF5 loaded!')
        print()

        all_classes.sort()
        all_classes_dir = {}
        idx = 1
        for single_class in all_classes:
            all_classes_dir[single_class] = self.complete_name(full_name='class_', idx=idx)
            idx += 1

        print('Creating basic directories tree...')
        os.mkdir(self.main_dir_name)
        os.mkdir(os.path.join(self.main_dir_name, self.img_dir_name))
        os.mkdir(os.path.join(self.main_dir_name, self.txt_dir_name))
        print('Basic directories tree created!')
        print()

        img_txt_idx = 1
        print('Saving data to directories...')
        for example_name, example_class_txt_img in all_data.items():
            example_class = example_class_txt_img[0][0]
            passive_name = all_classes_dir[example_class]
            if not os.path.exists(os.path.join(self.main_dir_name, self.txt_dir_name, passive_name)):
                os.mkdir(os.path.join(self.main_dir_name, self.txt_dir_name, passive_name))

            example_img = example_class_txt_img[0][2]
            example_img.save(os.path.join(self.main_dir_name, self.img_dir_name, self.complete_name(full_name='image_', idx=img_txt_idx, ext='jpeg')), 'JPEG')

            descriptions = ''
            for sample in example_class_txt_img:
                descriptions += sample[1]

            file = open(os.path.join(self.main_dir_name, self.txt_dir_name, passive_name, self.complete_name(full_name='image_', idx=img_txt_idx, ext='txt')), 'w')
            file.write(descriptions)
            file.close()

            img_txt_idx += 1
        print('Data saved in the new format!')

        print('Convertion finished!')