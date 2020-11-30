from txt2image_dataset2 import Text2ImageDataset2
from txt2image_dataset import Text2ImageDataset
from PIL import Image, ImageFont, ImageDraw
from models.gan_factory import gan_factory
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import Utils
from torch import nn

import datetime
import easydict
import torch
import yaml
import sys
import os
import gc

class Trainer(object):
    def __init__(self, dataset, split, lr, save_path, l1_coef, l2_coef, pre_trained_gen, pre_trained_disc,
                 val_pre_trained_gen, val_pre_trained_disc, batch_size, num_workers, epochs, dataset_paths,
                 arrangement, sampling ):

        with open('config.yaml', 'r') as f: # Wsteczna kompatybilnosc dla Text2ImageDataset
            config = yaml.safe_load(f)

        self.generator = torch.nn.DataParallel(gan_factory.generator_factory('gan').cuda())
        self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory('gan').cuda())

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(Utils.weights_init)

        self.val_pre_trained_gen = val_pre_trained_gen
        self.val_pre_trained_disc = val_pre_trained_disc
        self.arrangement = arrangement
        self.sampling = sampling

        if dataset == 'birds': # Wsteczna kompatybilnosc dla Text2ImageDataset
            self.dataset = Text2ImageDataset(config['birds_dataset_path'], split=split) # '...\Text2Image\datasets\ee285f-public\caltech_ucsd_birds\birds.hdf5'
        elif dataset == 'flowers': # Wsteczna kompatybilnosc dla Text2ImageDataset
            self.dataset = Text2ImageDataset(config['flowers_dataset_path'], split=split) # '...\Text2Image\datasets\ee285f-public\oxford_flowers\flowers.hdf5'
        elif dataset == 'live':
            self.dataset_dict = easydict.EasyDict(dataset_paths)
            self.dataset = Text2ImageDataset2(datasetFile = self.dataset_dict.datasetFile,
                                                imagesDir = self.dataset_dict.imagesDir,
                                                textDir = self.dataset_dict.textDir,
                                                split = split,
                                                arrangement = arrangement,
                                                sampling=sampling )
        else:
            print('Dataset not supported.')

        print('Images =', len(self.dataset))
        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) # shuffle=True - przetasowuje zbior danych w kazdej epoce

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path

        self.loss_filename_t = 'gd_loss_t.csv'
        self.loss_filename_v = 'gd_loss_v.csv'

    def train(self, cls):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        f = open(self.loss_filename_t, 'w')
        f.write('Epoka ; Dyskryminator ; Generator\n')
        f.close()

        f = open(self.loss_filename_v, 'w')
        f.write('Epoka ; Dyskryminator ; Generator\n')
        f.close()

        d_epoch_loss = g_epoch_loss = 0.0

        for epoch in range(self.num_epochs):
            real_epoch = epoch + 1
            iteration = 0

            dt = datetime.datetime.now()
            print('Epoch:', real_epoch, '/', self.num_epochs, 'started on', dt.date(), 'at', dt.time().replace(microsecond=0))

            for sample in self.data_loader:
                iteration += 1

                right_images = sample['right_images']
                right_embed = sample['right_embed']
                wrong_images = sample['wrong_images']

                right_images = Variable(right_images.float()).cuda()
                right_embed = Variable(right_embed.float()).cuda()
                wrong_images = Variable(wrong_images.float()).cuda()

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                wrong_loss = wrong_score = 0
                if cls:
                    outputs, _ = self.discriminator(wrong_images, right_embed)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss

                if cls:
                    d_loss = d_loss + wrong_loss

                d_loss.backward()
                self.optimD.step()

                d_epoch_loss += outputs.shape[0] * d_loss.item()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)
                _, activation_real = self.discriminator(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                g_loss = criterion(outputs, real_labels) + self.l2_coef * l2_loss(activation_fake[0], activation_real[0].detach()) + self.l1_coef * l1_loss(fake_images, right_images)

                g_loss.backward()
                self.optimG.step()

                g_epoch_loss += outputs.shape[0] * g_loss.item()

                iters_cnt = 100
                if iteration % iters_cnt == 0:
                    percentage = (iteration * 100.0) / len(self.data_loader)
                    print('Samples (iterations):', round(percentage, 2), '%, epochs:', real_epoch, '/', self.num_epochs)

            if real_epoch % 5 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, real_epoch)
                val_dataset = Text2ImageDataset2(datasetFile=self.dataset_dict.datasetFile,
                                                 imagesDir=self.dataset_dict.imagesDir,
                                                 textDir=self.dataset_dict.textDir,
                                                 split='valid',
                                                 arrangement=self.arrangement,
                                                 sampling=self.sampling )
                val_data_loader = DataLoader(val_dataset, batch_size=100, shuffle=False,
                                             num_workers=self.num_workers) # batch_size - zmienny
                self.validate(cls, val_dataset, val_data_loader, real_epoch)
                dt = datetime.datetime.now()
                print('Epoch', real_epoch, 'completed on', dt.date(), 'at', dt.time().replace(microsecond=0))
            else:
                dt = datetime.datetime.now()
                print('Epoch', real_epoch, 'completed on', dt.date(), 'at', dt.time().replace(microsecond=0))

            d_err = d_epoch_loss / len(self.dataset)
            g_err = g_epoch_loss / len(self.dataset)

            f = open(self.loss_filename_t, 'a')
            f.write(str(epoch) + " ; " + str(d_err) + " ; " + str(g_err) + "\n")
            f.close()

    def validate(self, cls, val_dataset, val_data_loader, epoch):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        generator = torch.nn.DataParallel(gan_factory.generator_factory('gan').cuda())
        generator.load_state_dict(torch.load(self.val_pre_trained_gen.replace("XXX", str(epoch))))

        discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory('gan').cuda())
        discriminator.load_state_dict(torch.load(self.val_pre_trained_disc.replace("XXX", str(epoch))))

        d_epoch_loss = g_epoch_loss = 0.0
        iteration = 0

        dt = datetime.datetime.now()
        print('Validating... Started on', dt.date(), 'at', dt.time().replace(microsecond=0))

        for sample in val_data_loader:
            iteration += 1

            right_images = sample['right_images']
            right_embed = sample['right_embed']
            wrong_images = sample['wrong_images']

            right_images = Variable(right_images.float()).cuda()
            right_embed = Variable(right_embed.float()).cuda()
            wrong_images = Variable(wrong_images.float()).cuda()

            real_labels = torch.ones(right_images.size(0))
            fake_labels = torch.zeros(right_images.size(0))

            smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

            real_labels = Variable(real_labels).cuda()
            smoothed_real_labels = Variable(smoothed_real_labels).cuda()
            fake_labels = Variable(fake_labels).cuda()

            outputs, activation_real = discriminator(right_images, right_embed)
            real_loss = criterion(outputs, smoothed_real_labels)
            real_score = outputs

            wrong_loss = wrong_score = 0
            if cls:
                outputs, _ = discriminator(wrong_images, right_embed)
                wrong_loss = criterion(outputs, fake_labels)
                wrong_score = outputs

            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = generator(right_embed, noise)
            outputs, _ = discriminator(fake_images, right_embed)
            fake_loss = criterion(outputs, fake_labels)
            fake_score = outputs

            d_loss = real_loss + fake_loss

            if cls:
                d_loss = d_loss + wrong_loss

            d_epoch_loss += outputs.shape[0] * d_loss.item()

            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = generator(right_embed, noise)
            outputs, activation_fake = discriminator(fake_images, right_embed)
            _, activation_real = discriminator(right_images, right_embed)

            activation_fake = torch.mean(activation_fake, 0)
            activation_real = torch.mean(activation_real, 0)

            g_loss = criterion(outputs, real_labels) + self.l2_coef * l2_loss(activation_fake[0], activation_real[0].detach()) + self.l1_coef * l1_loss(fake_images, right_images)
            # g_loss = BCE() + MSE(pomiedzy srednimi aktywacyjnymi dyskryminatora dla falszywych i prawdziwych obrazow) + MAE(pomiedzy obrazami)
            # g_loss = BCE() + bład średniokwadratowy wyznaczony pomiędzy średnimi aktywacjami dyskryminatora dla fałszywych i prawdziwych obrazów + bezwzględna róznica pomiędzy obrazami

            g_epoch_loss += outputs.shape[0] * g_loss.item()

            iters_cnt = 500
            if iteration % iters_cnt == 0:
                percentage = (iteration * 100.0) / len(val_data_loader)
                print('Samples (iterations):', round(percentage, 2), '%')

        dt = datetime.datetime.now()
        print('Validation completed on', dt.date(), 'at', dt.time().replace(microsecond=0))

        d_err = d_epoch_loss / len(val_dataset)
        g_err = g_epoch_loss / len(val_dataset)
        print('Discriminator loss:', d_err, '| generator loss:', g_err)

        f = open(self.loss_filename_v, 'a')
        f.write(str(epoch) + " ; " + str(d_err) + " ; " + str(g_err) + "\n")
        f.close()

        gc.collect()
        torch.cuda.empty_cache()

    def predict(self):
        self.print_info(len(self.dataset), self.batch_size, len(self.data_loader))

        best_res = -0.00000001
        best_idx = -1
        best_item = -1
        worst_res = 1.00000001
        worst_idx = sys.maxsize
        worst_item = sys.maxsize

        list_of_images = {}

        item = 0
        print()
        for sample in self.data_loader:
            item += 1

            right_images = sample['right_images']
            right_embed = sample['right_embed']
            txt = sample['txt']

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            right_images = Variable(right_images.float()).cuda()
            right_embed = Variable(right_embed.float()).cuda()

            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator(right_embed, noise)

            outputs, _ = self.discriminator(fake_images, right_embed)

            idx = 1
            for it in outputs:
                if it.item() > best_res:
                    best_idx = idx
                    best_res = it.item()
                    best_item = item
                if it.item() < worst_res:
                    worst_idx = idx
                    worst_res = it.item()
                    worst_item = item
                idx += 1

            image_no = 1
            data = ''
            for fimage, rimage, t, disc_out in zip(fake_images, right_images, txt, outputs):
                image_height = 64
                separeted_txt = self.separate_txt(t)
                img = self.get_concat_h(
                    Image.fromarray(rimage.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy()),
                    Image.fromarray(fimage.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy()),
                    len(separeted_txt) * 20)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("arial.ttf", 10)
                pxs = 0
                for single_txt in separeted_txt:
                    draw.text((0, image_height + pxs), single_txt, (255, 255, 255), font=font)
                    pxs += 10
                list_of_images[disc_out.item()] = img
                data = data + str(image_no) + '. ' + t
                image_no = image_no + 1

        print('The best sample_' + str(best_item) + '_image_' + str(best_idx) + ' has result: ' + str(best_res))
        print('The worst sample_' + str(worst_item) + '_image_' + str(worst_idx) + ' has result: ' + str(worst_res))

        tmp_list_of_images = list_of_images.items()
        sorted_list_of_images = sorted(tmp_list_of_images) # `reverse=True` - zamiast `for i in reversed(sorted_list_of_images)`
        idx = 1
        for i in reversed(sorted_list_of_images):
            i[1].save('results/' + self.save_path + '/image-' + str(idx) + '_' + str(i[0]) + '.jpeg')
            idx += 1

    def get_concat_h(self, im1, im2, add_space):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height + add_space))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def separate_txt(self, txt, n=26):
        return [txt[i:i+n] for i in range(0, len(txt), n)]

    def print_info(self, dataset, batch_size, data_loader):
        print('Dataset size:', dataset, '| batch size:', batch_size, '| number of samples:', data_loader)
        print('(' + str(dataset), '/', batch_size, '=', str(data_loader) + ')')