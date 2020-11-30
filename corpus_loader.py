from sklearn import feature_extraction

import tensorlayer as tl

import os


class CorpusLoader(object):

    def __init__(self, datasetFile, textDir):
        self.datasetFile = datasetFile
        self.textDir = textDir

    def load_folder_list(self, path=''):
        return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

    def TrainVocab(self):
        caption_dir = os.path.join(self.datasetFile, self.textDir)
        full_corpus_file = os.path.join(caption_dir, 'full_corpus.txt')
        captions_corpus = []

        if not os.path.isfile(full_corpus_file):
            print('File full_corpus.txt does not exists. Loading all files...')
            sentences_counter = 1
            images_counter = 1
            caption_sub_dir = self.load_folder_list(caption_dir)
            for sub_dir in caption_sub_dir:
                files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
                for file in files:
                    with open(os.path.join(sub_dir, file), "r") as f:
                        for i, line in enumerate(f):
                            line = line.replace('\n', '')
                            captions_corpus.append(line)
                            print('Image number', images_counter, ', sentence number', sentences_counter, end='\r')
                            sentences_counter = sentences_counter + 1
                    images_counter = images_counter + 1
            with open(full_corpus_file, 'w') as f:
                for item in captions_corpus:
                    f.write(item + '\n')
        else:
            print('File full_corpus.txt exists. Loading single file...')
            with open(full_corpus_file, "r") as f:
                captions_corpus = f.readlines()
                captions_corpus = [s.replace('\n', '') for s in captions_corpus]

        vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english', max_features=1024, binary=False)
        vectorizer.fit(raw_documents=captions_corpus)

        return vectorizer
