import math
import os
import cv2
import json
import random
import itertools
from time import time
from tqdm import tqdm
from copy import deepcopy
import shutil
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm_notebook

from albumentations import IAAAffine, PadIfNeeded, GaussNoise, Compose, CenterCrop

np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_printoptions(precision=10)

HEIGHT = 256
WIDTH = 256

def lock_all_seeds():
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_printoptions(precision=10)

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


def _save_Prep_files(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    shutil.copy('Preprocessing.py', os.path.join(model_checkpoints_folder, 'Preprocessing.py'))
    shutil.copy('data/dataset_wrapper.py', os.path.join(model_checkpoints_folder, 'dataset_wrapper.py'))




def _reshape_Identities(Identities):
    List_of_signs = []
    for _id, signs in enumerate(Identities):
        for sign in signs:
            sign['id'] = _id
            List_of_signs.append(sign)
    return List_of_signs




def Hard_train_test_split():
    from data.dataset_wrapper import Get_Signatures
    Identities = Get_Signatures()
    max_id = len(Identities)
    train_Identities, test_Identities = train_test_split(Identities, test_size=0.2, random_state=42)
    train_DS = _reshape_Identities(train_Identities)
    test_DS  = _reshape_Identities(test_Identities)

    return train_DS, test_DS, max_id


def Soft_train_test_split():
    from data.dataset_wrapper import Get_Signatures
    Identities = Get_Signatures()
    max_id = len(Identities)
    Signatures = _reshape_Identities(Identities)

    train_DS, test_DS = train_test_split(Signatures, test_size=0.2, random_state=42)

    return train_DS, test_DS, max_id


def _First_Aug(p=1, length=256):
    """
    Pad to square and do geometrical transforms
    """
    return Compose([
        PadIfNeeded(min_height=length, min_width=length, border_mode=cv2.BORDER_CONSTANT, value=255,
                    mask_value=0, always_apply=True),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, interpolation=1,
                         border_mode=cv2.BORDER_CONSTANT, value=255, p=0.9),
        IAAAffine(scale=1, translate_percent=None, translate_px=None, rotate=180.0,
                  shear=35, order=1, cval=255, mode='constant', always_apply=False, p=0.9)
    ], p=p)


def _resize_to_desired(img, max_height=HEIGHT, max_width=WIDTH, delta_aspect_r=0):
    """
    Resize image to max_height or max_width with changing aspect ratio
    """
    h, w = img.shape
    # aspect ratio
    new_aspect_ratio = w / h + delta_aspect_r
    if new_aspect_ratio > max_width/max_height:
        new_w = max_width
        new_h = int(new_w * 1 / new_aspect_ratio)
    else:
        new_h = max_height
        new_w = int(new_h * new_aspect_ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return img


def _Final_Aug(p=1, height=HEIGHT, width=WIDTH):
    return Compose([
        PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=255,
                    mask_value=0, always_apply=True),
        GaussNoise(var_limit=(0.01*255,0.1*255), mean=0, always_apply=False, p=0.5)
    ], p=p)


def Do_augs(data, height=HEIGHT, width=WIDTH):
    data=data.copy()
    img, _id = data['sign'], data['id']
    # Shear and rotations, shift
    img = _First_Aug(length=max(img.shape[:2]))(image=img)['image']
    # aspect ratio
    img = _resize_to_desired(img, max_height=height, max_width=width, delta_aspect_r=np.random.uniform(-0.3,0.3))
    # padding and Gauss noise
    new_img = _Final_Aug(height=height, width=width)(image=img)['image']
    #assert new_img.shape[0]==64 and new_img.shape[1]==128, f'wrong new h,w {new_img.shape[0]}x{new_img.shape[1]}'

    return new_img



class Train_Loader(torch.utils.data.Dataset):
    def __init__(self, Signs_list):
        super(Train_Loader, self).__init__()
        self.transform = transforms.ToTensor()
        self.Signs_list = Signs_list  #dict
        self.augmentations = Do_augs
    def __len__(self):
        return len(self.Signs_list)

    def __getitem__(self, i):
        data = self.Signs_list[i]#['sign']
        X = self.augmentations(data)
        X = self.transform(X)
        _id = self.Signs_list[i]['id']

        return X, _id


class Test_Loader(torch.utils.data.Dataset):
    def __init__(self, Signs_list):
        super(Test_Loader, self).__init__()
        self.final_padding = PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, border_mode=cv2.BORDER_CONSTANT, value=255,
                    mask_value=0, always_apply=True)
        self.transform = transforms.ToTensor()
        self.Signs_list = Signs_list

    def __len__(self):
        return len(self.Signs_list)

    def __getitem__(self, i):
        X = self.Signs_list[i]['sign']
        X = _resize_to_desired(X, delta_aspect_r=0)
        X = self.final_padding(image=X)['image']
        X = self.transform(X)
        _id = self.Signs_list[i]['id']

        return X, _id



def Get_dataloaders(batch_size=6, num_workers=0 , lock_seeds=True):
    print('loading dataset')
    if lock_seeds == True:
        lock_all_seeds()

    train_DS, test_DS, max_id = Soft_train_test_split()

    Training_DS = Train_Loader(train_DS)
    Train_batchloader = DataLoader(Training_DS, shuffle=True, batch_size=batch_size,
                                   num_workers=num_workers)

    Testing_DS = Test_Loader(test_DS)
    Test_batchloader = DataLoader(Testing_DS, shuffle=False, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=False)
    print(len(Train_batchloader), ' batches in train')
    print(len(Test_batchloader), ' batches in test')
    return Train_batchloader, Test_batchloader, max_id



class sCNN(object):
    def __init__(self, batch_size=10, num_workers=4, eval_factor=1):
        self.Train_batchloader, self.Test_batchloader, self.max_id = Get_dataloaders(batch_size=batch_size,
                                                                                     num_workers=num_workers)
        self.device = self._get_device()
        self.model = self.get_model().to(self.device)
        self.writer = None
        self.loss_func = nn.CrossEntropyLoss() #   nn.BCEWithLogitsLoss()
        self.eval_factor = eval_factor

        self.LOG = {'train': [], 'test': []}
        self.n_iter = 0


    def get_model(self):
        from models.models import sCNN
        print(f'max_id={self.max_id}')
        model = sCNN(num_classes = self.max_id)
        return model


    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device


    def _validate(self, model):
        model.eval()
        with torch.no_grad():
            Preds = []
            Y_trues_list = []
            epoch_loss, epoch_samples = 0, 0
            for X, Y_true in tqdm(self.Test_batchloader):
                emb, preds = model(X.cuda())
                loss = self.loss_func(preds, Y_true.cuda())
                # statistics
                epoch_loss += deepcopy(loss.item() * Y_true.shape[0])
                epoch_samples += deepcopy(Y_true.shape[0])

                soft = F.softmax(preds, dim=1)

                Preds.append(deepcopy(soft.argmax(1).detach().cpu().numpy().reshape(-1)))
                Y_trues_list.append(deepcopy(Y_true.detach().cpu().numpy().reshape(-1)))

            test_loss = epoch_loss / epoch_samples
            # scores
            Preds = np.concatenate(Preds).reshape(-1)
            Y_true = np.concatenate(Y_trues_list).reshape(-1)
            answ = (Preds == Y_true).astype(float)
            accuracy = answ.sum()/answ.shape[0]

            scores = dict(val_loss=test_loss,
                          accuracy=accuracy,
                          )

        to_writer = ['val_loss', 'accuracy']
        for score in to_writer:
            self.writer.add_scalar(score, scores[score], global_step=self.n_iter)

        model.train()
        return test_loss

    def train(self):
        eval_every_n_batches=(len(self.Train_batchloader)-1)*self.eval_factor

        model=self.model

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        self.writer = SummaryWriter()
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        print('checkpoints folder is ',model_checkpoints_folder)
        _save_Prep_files(model_checkpoints_folder)

        best_valid_loss = np.inf
        model.train()
        for epoch_counter in range(90000):
            epoch_loss, epoch_samples = 0, 0
            for i, (X, Y_true) in enumerate(tqdm(self.Train_batchloader)):
                optimizer.zero_grad()

                emb, preds = model(X.to(self.device))
                # forward
                loss = self.loss_func(preds, Y_true.to(self.device))

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=7)
                optimizer.step()
                # statistics
                epoch_loss += loss.item() * Y_true.shape[0]
                epoch_samples += Y_true.shape[0]

                if i % eval_every_n_batches == 0 and i!=0:
                    epoch_loss = epoch_loss / epoch_samples
                    self.writer.add_scalar('train_loss', epoch_loss, global_step=self.n_iter)
                    epoch_loss, epoch_samples = 0, 0

                    valid_loss = self._validate(model)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(model.state_dict(),
                                   os.path.join(model_checkpoints_folder, f'model_best.pth'))

                    self.n_iter += 1

            torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, f'model_last_epoch.pth'))

