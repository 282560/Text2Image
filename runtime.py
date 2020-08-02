from trainer import Trainer
import easydict

args = easydict.EasyDict({'type': 'gan', 
                        'lr': 0.001,
                        'l1_coef': 50,
                        'l2_coef': 100,
                        'cls': True,
                        'save_path':'./birds_cls_test',
                        'inference': True, # True - predict, False - train
                        'pre_trained_disc': 'checkpoints/birds_cls_test/disc_290.pth',
                        'pre_trained_gen': 'checkpoints/birds_cls_test/gen_290.pth',
                        'dataset': 'birds',
                        'split': 0,
                        'batch_size':16,
                        'num_workers':0, # Musi byc 0, aby Windows wspieral multiprocessing.
                        'epochs':300})

"""
args = easydict.EasyDict({'type': 'gan', 
                        'lr': 0.001,
                        'l1_coef': 50,
                        'l2_coef': 100,
                        'cls': True,
                        'save_path':'./flowers_cls_test',
                        'inference': True, # True - predict, False - train
                        'pre_trained_disc': 'checkpoints/flowers_cls/disc_190.pth',
                        'pre_trained_gen': 'checkpoints/flowers_cls/gen_190.pth',
                        'dataset': 'flowers',
                        'split': 0,
                        'batch_size':16,
                        'num_workers':0, # Musi byc 0, aby Windows wspieral multiprocessing.
                        'epochs':2})
"""

trainer = Trainer(type=args.type,
                  dataset=args.dataset,
                  split=args.split,
                  lr=args.lr,
                  save_path=args.save_path,
                  l1_coef=args.l1_coef,
                  l2_coef=args.l2_coef,
                  pre_trained_disc=args.pre_trained_disc,
                  pre_trained_gen=args.pre_trained_gen,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  epochs=args.epochs
                  )

if not args.inference:
    print('Inside of runtime/train.')
    trainer.train(args.cls)
else:
    print('Inside of runtime/predict.')
    trainer.predict()
