import numpy as np
import os
from os import listdir
from os.path import join
import torch
from torchvision import utils as vutils
import time
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.functional import normalize
from ev_utils import *

class NCaltech101:
    def __init__(self, root, augmentation=False, height=180, width=240):
        self.classes = listdir(root)

        self.files = []
        self.labels = []

        self.width = width
        self.height = height
        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)

        if self.augmentation:
            p = int((torch.rand(1)*0.5 + 0.5) * events.shape[0]) 
            events = self.sampler(events, p)
            events = random_shift_events_new(events, max_shift=20,resolution=(self.height, self.width))
            events = random_flip_events_along_x_new(events, resolution=(self.height, self.width))

        count_vox = voxel_represent_count(events[:,0], events[:,1], events[:,2], 8, self.height, self.width)
        avg_vox = voxel_represent_avgt(events[:,0], events[:,1], events[:,2], 9, self.height, self.width)
        Range_Img = count_vox
        Residual_Img = residual_img(avg_vox)

        # fxy = Fxy(events[:,0], events[:,1], events[:,2], events[:,3], 8, [240,180])
        # fxy = F.interpolate(fxy.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=True).squeeze(0)
        # Range_Img = fxy

        fus = torch.cat([Range_Img,Residual_Img])  
        fus = F.interpolate(fus.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=True).squeeze(0)

        events = torch.from_numpy(events)
        # 时间归0
        events[:,2] = events[:,2]-torch.min(events[:,2])
        factor = 200
        if torch.max(events[:,2])>0:
            events[:,2] = events[:,2]/torch.max(events[:,2]) * factor
        assert not torch.any(torch.isnan(events))

        # events = self.sampler(events, 8192)

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

