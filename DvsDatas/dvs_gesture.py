import torch
import torch.nn as nn
import numpy as np
import os
from os import listdir
import random
import torch.nn.functional as F
from DvsDatas.ev_utils import *



class Gusture:
    def __init__(self, root, height=128, width=128, augmentation=False, mode='training'):
        """
        Creates an iterator over the N_Caltech101 dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        """
        if mode == 'training':
            mode = 'train'
        elif mode == 'testing':
            mode = 'test'
        if mode == 'validation':
            mode = 'test'
        self.root = os.path.join(root, mode)
        self.width = width
        self.height = height
        self.augmentation = augmentation

        self.files = [os.path.join(self.root, f) for f in listdir(self.root)]

        print(len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        events, label = np.load(self.files[idx], allow_pickle=True)
        events = events.astype(np.float32)
        events[events[:, -1] == 0, -1] = -1
        
        if self.augmentation:
            p = int((torch.rand(1)*0.5 + 0.5) * events.shape[0]) 
            events = self.sampler(events, p)
            events = random_shift_events_new(events, max_shift=8,resolution=(self.height, self.width))
            events = random_flip_events_along_x_new(events, resolution=(self.height, self.width))

        count_vox = voxel_represent_count(events[:,0], events[:,1], events[:,2], 8, self.height, self.width)
        avg_vox = voxel_represent_avgt(events[:,0], events[:,1], events[:,2], 9, self.height, self.width)
        
        Range_Img = count_vox
        Residual_Img = residual_img(avg_vox)

        # fxy = Fxy(events[:,0], events[:,1], events[:,2], events[:,3], 8, [128,128])
        # Range_Img = fxy

        fus = torch.cat([Range_Img,Residual_Img])  
        fus = F.interpolate(fus.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=True).squeeze(0)
        # fus = F.interpolate(fus.unsqueeze(0).unsqueeze(0), size=(fus.shape[0],224,224), mode='trilinear', align_corners=True).squeeze()

        events = torch.from_numpy(events)
        # 时间归0
        events[:,2] = events[:,2]-torch.min(events[:,2])
        factor = 200
        if torch.max(events[:,2])>0:
            events[:,2] = events[:,2]/torch.max(events[:,2]) * factor
        assert not torch.any(torch.isnan(events))
        assert not torch.any(torch.isnan(fus))

        return fus.float(), events, label
    
    def sampler(self, events, nsample):
        # events number
        N = events.shape[0]
        if N>nsample:     
            # random select nsample index
            indices = torch.randperm(N)[:nsample]
            # get corresponding events
            sampled_events = events[indices]
        
        elif N<nsample:
            expend_factor = nsample//N
            sampled_events = events.repeat(expend_factor,1)
            num_copies = nsample%N
            indices_copy = torch.randperm(N)[:num_copies]
            sampled_events = torch.cat((sampled_events,events[indices_copy]),0)

        else:
            sampled_events = events

        return sampled_events
    

