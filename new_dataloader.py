from torch.utils.data import Dataset, DataLoader
from PIL import Image

import tensorlayer as tl
import numpy as np

import scipy.misc
import string
import random
import gensim
import scipy
import torch
import nltk
import time
import math
import os
import re

import matplotlib.pyplot as plt

class Text2ImageDataset(Dataset):

    def __init__(self, datasetFile, imagesDir, textDir):
        self.datasetFile = datasetFile
        self.imagesDir = imagesDir
        self.textDir = textDir

        self.number_of_images = self.__len__()

        self.images_classes = {}
        self.assign_classes()

    def __len__(self):
        path_to_images = os.path.join(self.datasetFile, self.imagesDir)
        path, dirs, files = next(os.walk(path_to_images))
        length = len(files)
        return length

    # Rev.0001 - Obiekty, ktore sa oficjalnie identycznymi jak te z poprzedniej implementacji
    def __getitem__(self, idx):
        # Numer obrazu:
        image_index = math.floor(idx / 5) + 1 # W sytuacji, gdy wystepuje 5 opisow na obraz

        # Numer opisu w sytuacji, gdy wystepuje 5 opisow na obraz:
        caption_index = idx % 5

        img_dir = os.path.join(self.datasetFile, self.imagesDir)
        name = self.complete_name(idx=idx, ext='jpeg')

        right_image = Image.open(os.path.join(img_dir, name)).resize((64, 64)) # Rev.0001 - right_image
        right_image = self.validate_image(right_image)                         # Rev.0001 - right_image

        wrong_image_name = self.find_wrong_image_name(name, self.images_classes)           # Rev.0001 - wrong_image
        wrong_image = Image.open(os.path.join(img_dir, wrong_image_name)).resize((64, 64)) # Rev.0001 - wrong_image
        wrong_image = self.validate_image(wrong_image)                                     # Rev.0001 - wrong_image

        caption_name = self.complete_name(idx=idx, ext='txt')
        caption_dir = os.path.join(self.datasetFile, self.textDir)
        caption_sub_dir = os.path.join(caption_dir, self.images_classes[name], caption_name)

        print('\nLokalizacja:', caption_sub_dir, '\n')

        sentences = []
        txt = ''
        with open(caption_sub_dir, "r") as file:
            content = file.readlines()
        content = [x.strip() for x in content]
        for line in content:
            sentences.append(line.split())

        model = gensim.models.Word2Vec(size=1024, iter=50, min_count=4)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        word_vectors = model.wv
        test_vector = word_vectors.word_vec('yellow')
        print('Vocabulary:', word_vectors.vocab, '\n')
        print('Ilosc kluczy:', len(word_vectors.vocab.keys()), '\n')
        print('word_vectors.word_vec(\'yellow\'):', '\n', test_vector, '\n')
        print('len(test_vector):', len(test_vector), '\n')
        print('------------------')
        print('model.corpus_total_words:', model.corpus_total_words)
        print('model.batch_words:', model.batch_words)
        print('model.min_count:', model.min_count)
        #print('', model)
        print('------------------')
        print('word_vectors.vector_size:', word_vectors.vector_size)
        print('word_vectors.index2entity:', word_vectors.index2entity)
        print('word_vectors.index2word:', word_vectors.index2word)
        print('------------------')

        exit()

        # Rev.0001 - right_image, wrong_image
        sample = {
            'right_images': torch.FloatTensor(right_image),
            #'right_embed': torch.FloatTensor(right_embed),
            'wrong_images': torch.FloatTensor(wrong_image),
            #'inter_embed': torch.FloatTensor(inter_embed), # NIEUZYWANE
            'txt': txt }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5) # Rev.0001 - right_image
        sample['wrong_images'] = sample['wrong_images'].sub_(127.5).div_(127.5) # Rev.0001 - wrong_image

        return sample

########################################################################################################################

    # Rev.0001 - Obiekty, ktore sa oficjalnie identycznymi jak te z poprzedniej implementacji
    def LoadingData(self):
        cwd = os.getcwd()
        VOC_FIR = cwd + '/vocab.txt'

        files_counter_list = []

        caption_dir = os.path.join(self.datasetFile, self.textDir)
        caption_sub_dir = self.load_folder_list(caption_dir)

        captions_dict = {}
        '''
            `captions_dict` to zbior zdan dla kazdego obrazka, na przyklad dla pierwszego obrazka:
            [
                'the petals of the flower are pink in color and have a yellow center ',
                'this flower is pink and white in color  with petals that are multi colored ',
                'the geographical shapes of the bright purple petals set off the orange stamen and filament and the cross shaped stigma is beautiful ',
                'the purple petals have shades of white with white anther and filament',
                'this flower has large pink petals and a white stigma in the center',
                'this flower has petals that are pink and has a yellow stamen',
                'a flower with short and wide petals that is light purple ',
                'this flower has small pink petals with a yellow center ',
                'this flower has large rounded pink petals with curved edges and purple veins ',
                'this flower has purple petals as well as a white stamen '
            ]
        '''
        processed_capts = []
        '''
            `processed_capts` to zbior wszystkich zdan zebranych w jednym obiekcie, na przyklad pierwsza wartosc:
                ['<S>', 'the', 'petals', 'of', 'the', 'flower', 'are', 'pink', 'in', 'color', 'and', 'have', 'a', 'yellow', 'center', '</S>']
            na przyklad dziesiata wartosc:
                ['<S>', 'this', 'flower', 'has', 'purple', 'petals', 'as', 'well', 'as', 'a', 'white', 'stamen', '</S>']
            i jedenasta:
                ['<S>', 'this', 'white', 'and', 'purple', 'flower', 'has', 'fragile', 'petals', 'and', 'soft', 'stamens', '</S>']
            (...)
        '''

        for sub_dir in caption_sub_dir:
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')

            files_counter_list.append(len(files))

            for i, f in enumerate(files):
                file_dir = os.path.join(sub_dir, f)
                key = int(re.findall('\d+', f)[0])
                t = open(file_dir, 'r')
                lines = []
                for line in t:
                    line = self.preprocess_caption(line)
                    lines.append(line)
                    processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
                assert len(lines) == 10, "Every flower image have 10 captions"
                captions_dict[key] = lines

        self.show_files_counting(files_counter_list)

        if not os.path.isfile('vocab.txt'):
            _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
        else:
            print('WARNING: vocab.txt already exists!')

        vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")
        '''
            `vocab` - obiekt klasy Vocabulary stworzony z zadanego slownictwa wraz z konwersja slowo-id, id-slowo.
        '''

        captions_ids = []
        ids = captions_dict.items()
        for key, value in ids:
            for v in value:
                captions_ids.append( [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id] )
                '''
                    `v` - pojedyncze zdanie na przyklad:
                        the petals of the flower are pink in color and have a yellow center
                    `captions_ids` - zbior wektorow, gdzie kazdy ma na koncu `vocab.end_id`, na przyklad
                        [
                            [11, 4, 19, 11, 3, 8, 16, 15, 20, 5, 43, 10, 13, 25, 2],
                            [6, 3, 17, 16, 5, 14, 15, 20, 9, 4, 12, 8, 125, 50, 2],
                            [11, 3082, 406, 19, 11, 31, 18, 4, 277, 302, 11, 24, 21, 5, 122, 5, 11, 1074, 26, 32, 17, 161, 2],
                            [11, 18, 4, 43, 151, 19, 14, 9, 14, 72, 5, 122, 2]
                            (...)
                        ]
                '''
        captions_ids = np.asarray(captions_ids)

        '''
            # Sprawdzenie...
            img_capt = captions_dict[1][1]
            print("img_capt: %s" % img_capt)
            print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
            img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]  # img_capt.split(' ')]
            print("img_capt_ids: %s" % img_capt_ids)
            print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])
        '''

        img_dir = os.path.join(self.datasetFile, self.imagesDir)
        imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpeg'))

        print('%d images found, start loading and resizing...' % len(imgs_title_list))
        s = time.time()

        images = []
        print('\n')
        for name in imgs_title_list:
            print('Getting image:', os.path.join(img_dir, name))
            img_raw = scipy.misc.imread(os.path.join(img_dir, name))
            img_raw = np.asarray(img_raw)
            img = tl.prepro.imresize(img_raw, size=[64,64]) # [64, 64, 3]
            img = img.astype(np.float32)
            images.append(img)
            print('Image', name, 'loaded', end='\r')
        print('\nLoading and resizing took %ss' % (time.time() - s))

        exit()

        n_images = len(captions_dict)
        n_captions = len(captions_ids)
        n_captions_per_image = len(lines) # Local variable 'lines' might be referenced before assignment

        print('Images:', n_images, '| Captions:', n_captions, '| Captions per image:', n_captions_per_image)

        captions_ids_train, captions_ids_test = captions_ids[ : 8000 * n_captions_per_image], captions_ids[8000 * n_captions_per_image : ]
        images_train, images_test = images[ : 8000], images[8000 : ]

        n_images_train = len(images_train)
        n_images_test = len(images_test)

        n_captions_train = len(captions_ids_train)
        n_captions_test = len(captions_ids_test)

        print('Images train:', n_images_train, '| Captions train:', n_captions_train)
        print('Images test:', n_images_test, '| Captions test:', n_captions_test)

########################################################################################################################

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

########################################################################################################################

    def load_folder_list(self, path=''):
        return [os.path.join(path,o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]

########################################################################################################################

    def preprocess_caption(self, line):
        prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
        prep_line = prep_line.replace('-', ' ')
        return prep_line

########################################################################################################################

    def show_files_counting(self, files_counter_list):
        sum_list = sum(files_counter_list)
        avg = sum_list / len(files_counter_list)
        print('Number of files:', sum_list, '| Average number of txt files per class:', str(round(avg, 2)), '\n')

########################################################################################################################

    def complete_name(self, idx, ext):
        num = idx
        positions = []
        while num != 0:
            positions.append(num % 10)
            num = num // 10
        fullfill = ((5 - len(positions)) * '0')
        for i in reversed(positions):
            fullfill = fullfill + str(i)
        return 'image_' + fullfill + '.' + ext

########################################################################################################################

    def find_wrong_image_name(self, img_name, images_classes):
        img_class = images_classes[img_name]
        while(True):
            rnd = random.randint(1, self.number_of_images)
            wrong_img_name = self.complete_name(idx=rnd, ext='jpeg')
            wrong_img_class = images_classes[wrong_img_name]
            if (img_name != wrong_img_name) and (img_class != wrong_img_class):
                return wrong_img_name

########################################################################################################################

    def assign_classes(self):
        caption_dir = os.path.join(self.datasetFile, self.textDir)
        caption_sub_dir = self.load_folder_list(caption_dir)

        for sub_dir in caption_sub_dir:
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')

            for file in files:
                pre, ext = os.path.splitext(file)
                self.images_classes[pre + '.jpeg'] = os.path.basename(os.path.normpath(sub_dir))

        self.images_classes = dict(sorted(self.images_classes.items())) # Posortowane po kluczu
        #return dict(sorted(images_classes.items())) # Posortowane po kluczu