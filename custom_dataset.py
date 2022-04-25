import torch
import random
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pickle
import os
from datetime import datetime
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage
import matplotlib.pyplot as plt

# torch.multiprocessing.set_start_method('spawn')

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Domain randomization
img_transform = iaa.Sequential([
    iaa.LinearContrast((0.95, 1.05), per_channel=0.25), 
    iaa.Add((-10, 10), per_channel=False),
    #iaa.GammaContrast((0.95, 1.05)),
    #iaa.GaussianBlur(sigma=(0.0, 0.6)),
    #iaa.MultiplySaturation((0.95, 1.05)),
    #iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
    iaa.flip.Flipud(0.5),
    sometimes(iaa.Affine(
        scale = {"x": (0.7, 1.3), "y": (0.7, 1.3)},
        rotate=(-30, 30),
        shear=(-30, 30)
        ))
    ], random_order=True)

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False, single=False):
    if not single:
        U.unsqueeze_(1).unsqueeze_(2)
        V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1), torch.transpose(Y, 0, 1)
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    if normalize_dist:
        return normalize(G).double()
    return G.double()

def vis_gauss(gaussians):
    gaussians = gaussians.cpu().numpy()
    h1 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)

def bimodal_gauss(G1, G2, normalize=False):
    bimodal = torch.max(G1, G2)
    if normalize:
        return normalize(bimodal)
    return bimodal

class ConditioningDataset(Dataset):
    def __init__(self, img_folder, labels_folder, img_height, img_width, gauss_sigma=8, augment=True, finetune=False):
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = transform
        self.img_transform = img_transform
        self.augment = augment
        self.finetune = finetune

        self.imgs = []
        self.labels = []
        for i in range(len(os.listdir(labels_folder))):
            label_path = os.path.join(labels_folder, '%05d.npy'%i)
            if not  os.path.exists(label_path):
                continue
            label = np.load(label_path)
            if len(label) > 0:
                label[:,0] = np.clip(label[:, 0], 0, self.img_width-1)
                label[:,1] = np.clip(label[:, 1], 0, self.img_height-1)
                img_path = os.path.join(img_folder, '%05d.png'%i)
                img_save = cv2.imread(img_path)
                self.imgs.append(img_save)
                self.labels.append(label)

    def __getitem__(self, index):
        keypoints = self.labels[index]
        img = (self.imgs[index]).copy()
        if not self.finetune:
            keypoints = keypoints[:,::-1]
        kpts = KeypointsOnImage.from_xy_array(keypoints, shape=img.shape)
        if self.augment:
            img, labels = self.img_transform(image=img, keypoints=kpts)
        else:
            labels = kpts
        img = img.copy()
        img = self.transform(img)
        labels_np = []
        for l in labels:
            labels_np.append([l.x,l.y])
        labels = torch.from_numpy(np.array(labels_np, dtype=np.int32))
        if not self.finetune:
            condition_point = labels[np.random.randint(len(labels))]
        else:
            condition_point = labels[0]
        cond_gauss = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, condition_point[0], condition_point[1], single=True)
        cond_gauss = torch.unsqueeze(cond_gauss, 0)
        img = img.detach().cpu().numpy().transpose((1,2,0))
        img[:,:,0] = cond_gauss.detach().cpu().numpy().transpose((1,2,0))[:,:,0]
        img = self.transform(img)
        if self.finetune:
            U = labels[1:,0]
            V = labels[1:,1]
            gaussians = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U, V)
            mm_gauss = gaussians[0]
            for i in range(1, len(gaussians)):
                mm_gauss = bimodal_gauss(mm_gauss, gaussians[i])
            mm_gauss.unsqueeze_(0)
            mm_gauss = torch.tensor(np.concatenate((mm_gauss, mm_gauss, mm_gauss)))
            return img, mm_gauss
        else:
            
            return img
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    IMG_WIDTH = 100
    IMG_HEIGHT = 100
    GAUSS_SIGMA = 8
    test_dataset = ConditioningDataset('/home/vainavi/mae/train_sets/one_img/train/images',
                           '/home/vainavi/mae/train_sets/one_img/train/annots',
                           IMG_HEIGHT, IMG_WIDTH, gauss_sigma=GAUSS_SIGMA, finetune=False)
    # combined = test_dataset[0]
    # combined = combined.detach().cpu().numpy()
    # img = combined.transpose((1, 2, 0))
    # plt.imsave("test.png", img)

    if not os.path.exists("test_dataset"):
        os.mkdir("test_dataset")
    for i in range(len(test_dataset)):
        combined = test_dataset[i]
        combined = combined.cpu().numpy()
        img = combined.transpose((1, 2, 0))
        plt.imsave("test_dataset/%d.png"%i, img)