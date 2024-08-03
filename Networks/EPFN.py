import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import torch_scatter
from functools import reduce

# Swish功能
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.sw = Swish()

    def forward(self, inputs, unq_inv):
        torch.backends.cudnn.enabled = False
        x = self.linear(inputs)
        x = self.norm(x)
        
        # x = F.relu(x)

        # Swish代替relu激活函数
        x = self.sw(x)

        torch.backends.cudnn.enabled = True

        # max pooling
        feat_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        x_max = feat_max[unq_inv]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max], dim=1)
            return x_concatenated


class PillarNet(nn.Module):

    def __init__(self,
                 num_input_features=4,
                 voxel_size=(3,3,1),
                 pc_range=(346,260,200)):
        super().__init__()
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

    def forward(self, points):

        device = points.device
        dtype = points.dtype

        # discard out of range points
        grid_size = np.ceil(self.pc_range/self.voxel_size).astype(np.int64)  # x,  y

        voxel_size = torch.from_numpy(self.voxel_size).to(device)
        pc_range = torch.from_numpy(self.pc_range).to(device)

        points_coords = points[:, 1:4] // voxel_size.view(-1, 3)   # x, y, z
        points_coords = points_coords.long()
        batch_idx = points[:, 0:1].long()

        points_index = torch.cat((batch_idx, points_coords[:, :2]), dim=1)
        unq, unq_inv = torch.unique(points_index, return_inverse=True, dim=0)
        unq = unq.int()

        points_mean_scatter = torch_scatter.scatter_mean(points[:, 1:4], unq_inv, dim=0)
        # points_std_scatter = torch_scatter.scatter_std(points[:, 3:4], unq_inv, dim=0)

        f_cluster = points[:, 1:4] - points_mean_scatter[unq_inv]
        # f_cluster = points_mean_scatter[unq_inv]

        # Find distance of x, y, and z from pillar center
        f_center = points[:, 1:3] - (points_coords[:, :2] * voxel_size[:2].unsqueeze(0) +
                                     (voxel_size[:2]-1).unsqueeze(0) / 2)

        # Combine together feature decorations
        features = torch.cat([points[:, 1:], f_cluster, f_center], dim=-1)

        # features[:,16] = features[:,15] - features[:,2]
        # features[:,17:18] = points_std_scatter[unq_inv]

        # norm
        # features[:,0:2] = self.normalize_point_cloud(features[:,0:2])
        # features[:,2:3] = self.normalize_point_cloud(features[:,2:3])

        # (x,y,t,p,x_c,y_c,t_c,x_d.y_d)
        features[:,0] /= self.pc_range[0]
        features[:,1] /= self.pc_range[1]
        features[:,2] /= self.pc_range[2]
        features[:,4] /= self.voxel_size[0]
        features[:,5] /= self.voxel_size[1]
        features[:,6] /= self.pc_range[2]
        features[:,7] /= self.voxel_size[0]
        features[:,8] /= self.voxel_size[1]
        # print(features[100])
        # features = features[:,[0,1,2,3]]

        return features[:,:4], unq[:, [0, 2, 1]], unq_inv, grid_size[[1, 0]], batch_idx

    def normalize_point_cloud(self, pc):
        centroid = torch.mean(pc, axis=0) # 求取点云的中心
        pc = pc - centroid # 将点云中心置于原点 (0, 0, 0)
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1))) # 求取长轴的的长度
        pc_normalized = pc / m # 依据长轴将点云归一化到 (-1, 1)
        return pc_normalized  # centroid: 点云中心, m: 长轴长度, centroid和m可用于keypoints的计算


class EventPillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features,
        num_filters,
        voxel_size,
        pc_range,
        norm_cfg=None,
    ):

        super().__init__()
        assert len(num_filters) > 0
        # num_input_features = 4

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers1 = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True

            pfn_layers1.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers1 = nn.ModuleList(pfn_layers1)

        self.feature_output_dim = num_filters[-1]

        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

        self.voxelization = PillarNet(num_input_features, voxel_size, pc_range)

        self.attention_block = GC(32,4)

        self.att = nn.Sequential(
            nn.Linear(3,6),
            nn.ReLU(),
            nn.Linear(6,3),
            nn.Sigmoid()
        )

    def forward(self, fus, points):
        features, coords, unq_inv, grid_size, batch_idx = self.voxelization(points)
        # Forward pass through PFNLayers
        features1 = features

        for pfn in self.pfn_layers1:
            features1 = pfn(features1, unq_inv)  # num_points, dim_feat

        # feat_mean1 = torch_scatter.scatter_mean(features1, unq_inv, dim=0)
        feat_max1 = torch_scatter.scatter_max(features1, unq_inv, dim=0)[0]
        # feat_sum1 = torch_scatter.scatter_sum(features1, unq_inv, dim=0)
        # feat = torch.cat([feat_mean1, feat_max1], dim=1)
       
        # Attention Block
        # feat_max1 = self.attention_block(feat_max1)
        # feat_max2 = self.attention_block(feat_max2)
        # feat_max3 = self.attention_block(feat_max3)

        feat = feat_max1

        import spconv.pytorch as spconv
        batch_size = len(torch.unique(coords[:, 0]))
        feat = spconv.SparseConvTensor(feat, coords, grid_size,batch_size)
        feat = feat.dense()
        # feat = F.interpolate(feat.unsqueeze(1), size=(feat.shape[1],224,224), mode='trilinear', align_corners=True).squeeze(1)
        # feat = F.interpolate(feat, size=(224//self.voxel_size[0],224//self.voxel_size[1]), mode='bilinear', align_corners=True)
        feat = F.interpolate(feat, size=(224,224), mode='bilinear', align_corners=True)
            
        return feat                                                  


class GC(torch.nn.Module):
    def __init__(self,in_channel,ratio):
        super(GC, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel,1,kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channel,in_channel//ratio,kernel_size=1)
        self.conv3 = torch.nn.Conv2d(in_channel//ratio,in_channel,kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.ln = torch.nn.LayerNorm([in_channel//ratio,1,1])
        self.relu = torch.nn.ReLU()

    def forward(self,input):
        b,c,w,h = input.shape
        x = self.conv1(input).view([b,1,w*h]).permute(0,2,1)
        x = self.softmax(x)
        i = input.view([b,c,w*h])
        x = torch.bmm(i,x).view([b,c,1,1])
        x = self.conv2(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x+input
    