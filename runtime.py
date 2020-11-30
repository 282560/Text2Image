from trainer import Trainer

import argparse
import easydict
import os


parser = argparse.ArgumentParser(prog='runtime.py',
    usage='python %(prog)s [-h] operation dataset learning_rate batch_size epochs',
    description='Text to image synthesis using GAN.',
    epilog='example of use: `python %(prog)s train flowers 0.0002 100 1000`')
parser.add_argument('operation',
    action='store',
    type=str,
    help='Program main operation name (train or test).')
parser.add_argument('dataset',
    action='store',
    type=str,
    help='Processed dataset name.')
parser.add_argument('learning_rate',
    action='store',
    type=float,
    help='Learning rate value (0.0002 recommended).')
parser.add_argument('batch_size',
    action='store',
    type=int,
    help='Batch size.')
parser.add_argument('epochs',
    action='store',
    type=int,
    help='Number of epochs.')
args = parser.parse_args()

operation = args.operation
ds_name = args.dataset
trainer = inference = cls = ''
learning_rate = args.learning_rate

paths = easydict.EasyDict({})
arrangement = easydict.EasyDict({})
sampling = easydict.EasyDict({})

if ds_name == 'flowers':
    paths = easydict.EasyDict({
        'datasetFile': os.path.join('datasets', '102flowers'),
        'imagesDir': '102flowers',
        'textDir': 'text_c10'
    })
    arrangement = easydict.EasyDict({
        'train': 5879,
        'valid': 1155,
        'test': 1155
    }) # 8 189
    sampling = easydict.EasyDict({
        'train': 10,
        'valid': 10,
        'test': 10
    })
elif ds_name == 'birds':
    paths = easydict.EasyDict({
        'datasetFile': os.path.join('datasets', 'caltech_ucsd_birds'),
        'imagesDir': 'caltech_ucsd_birds',
        'textDir': 'text_c10'
    })
    arrangement = easydict.EasyDict({
        'train': 7792,
        'valid': 1500,
        'test': 1500
    }) # 10 792
    sampling = easydict.EasyDict({
        'train': 5,
        'valid': 5,
        'test': 5
    })
elif ds_name == 'three_flowers':
    paths = easydict.EasyDict({
        'datasetFile': os.path.join('datasets', 'three_flowers'),
        'imagesDir': 'three_flowers',
        'textDir': 'text_c10'
    })
    arrangement = easydict.EasyDict({
        'train': 54,
        'valid': 3,
        'test': 3
    }) # 60
    sampling = easydict.EasyDict({
        'train': 12,
        'valid': 12,
        'test': 12
    })
elif ds_name == 'three_fruits':
    paths = easydict.EasyDict({
        'datasetFile': os.path.join('datasets', 'three_fruits'),
        'imagesDir': 'three_fruits',
        'textDir': 'text_c10'
    })
    arrangement = easydict.EasyDict({
        'train': 54,
        'valid': 3,
        'test': 3
    }) # 60
    sampling = easydict.EasyDict({
        'train': 12,
        'valid': 12,
        'test': 12
    })
elif ds_name == 'three_birds':
    paths = easydict.EasyDict({
        'datasetFile': os.path.join('datasets', 'three_birds'),
        'imagesDir': 'three_birds',
        'textDir': 'text_c10'
    })
    arrangement = easydict.EasyDict({
        'train': 54,
        'valid': 3,
        'test': 3
    }) # 60
    sampling = easydict.EasyDict({
        'train': 20,
        'valid': 20,
        'test': 20
    })

if operation == 'train':
    trainer = Trainer(
        dataset='live', # 'live' | 'flowers' | 'birds'
        split=operation,
        lr=learning_rate,
        save_path='./' + ds_name + '_cls_test',
        l1_coef=50,
        l2_coef=100,
        pre_trained_gen=False, # Ustawione na False nie wczytuje checkpoint'ow
        pre_trained_disc=False, # Ustawione na False nie wczytuje checkpoint'ow
        val_pre_trained_gen='checkpoints/' + ds_name + '_cls_test/gen_XXX.pth',   # 'XXX' jest wymagane, poniewaz liczba epok jest dobierana dynamicznie
        val_pre_trained_disc='checkpoints/' + ds_name + '_cls_test/disc_XXX.pth', # 'XXX' jest wymagane, poniewaz liczba epok jest dobierana dynamicznie
        batch_size=args.batch_size, # 100
        num_workers=0, # Musi byc 0, aby Windows wspieral multiprocessing.
        epochs=args.epochs, # 1000
        dataset_paths=paths,
        arrangement=arrangement,
        sampling=sampling )
    inference = False # True - predict | False - train
    cls = True
elif operation == 'valid':
    print('Validation is done inside training. You cannot do it on your own. Change operation to \'train\' or \'test\' (not \'valid\').')
    exit()
elif operation == 'test':
    trainer = Trainer(
        dataset='live', # 'live' | 'flowers' | 'birds'
        split=operation,
        lr=learning_rate,
        save_path='./' + ds_name + '_cls_test',
        l1_coef=50,
        l2_coef=100,
        pre_trained_gen='checkpoints/' + ds_name + '_cls_test/gen_' + str(args.epochs) + '.pth', # Ustawione na False nie wczytuje checkpoint'ow
        pre_trained_disc='checkpoints/' + ds_name + '_cls_test/disc_' + str(args.epochs) + '.pth', # Ustawione na False nie wczytuje checkpoint'ow
        val_pre_trained_gen=False,
        val_pre_trained_disc=False,
        batch_size=args.batch_size, # 3
        num_workers=0, # Musi byc 0, aby Windows wspieral multiprocessing.
        epochs=args.epochs, # 600
        dataset_paths=paths,
        arrangement=arrangement,
        sampling=sampling )
    inference = True # True - predict | False - train
    cls = True

if not inference:
    print('Inference=' + str(inference) + ', starting training.')
    trainer.train(cls)
else:
    print('Inference=' + str(inference) + ', starting prediction.')
    trainer.predict()