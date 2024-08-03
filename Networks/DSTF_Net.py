import torch
import torch.nn as nn
from EMFN import EventMapFeatureNet
from EPFN import EventPillarFeatureNet

class get_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = Resnet(n=2)
        self.emfn = EventMapFeatureNet(nclasses=2,range_channel=8, n_input_scans=8, 
                                                batch_size=12, height=224, width=224, num_batch=None, point_refine=None)
        self.epfn = EventPillarFeatureNet(num_input_features=4,
                                num_filters=(16,32),
                                voxel_size=(1,1,1),
                                pc_range=(120,100,200),)
        self.att = SELay()
        # self.att = GC()
        
    def forward(self, fus, events):
        x = self.emfn(fus)
        y = self.epfn(events, events)
        # x = self.att(x)
        # y = self.att(y)
        x = torch.cat([x,y], dim=1)
        x = self.att(x)
        x = self.fc(x)
        return x


from torchvision.models.resnet import resnet50 ,resnet34, resnet18    
class Resnet(nn.Module):
    def __init__(self, n=101) -> None:
        super().__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, n)
    def forward(self, input):
        x = self.resnet(input)
        return x


class SELay(nn.Module):
    def __init__(self, c=[64,32]):
        super(SELay, self).__init__()
        self.AAP = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c[0], c[1])
        self.fc2 = nn.Linear(c[1], c[0])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        a = self.AAP(x)
        a = self.fc1(a.squeeze(2).squeeze(2))
        a = self.relu(a)
        a = self.fc2(a)
        a = self.sigmoid(a)
        a = a.unsqueeze(2).unsqueeze(3)
        x = x*a
        return x
    

class GC(torch.nn.Module):
    def __init__(self,in_channel=64,ratio=2):
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
    

if __name__ == "__main__":

    model = get_model()
    fus = torch.randn(1, 16, 224, 224)
    events = torch.randint(0,180,(8192,5))
    events[:,0]=0
    # output = model(input)
    # print(output.shape)

    from thop import profile
    flops, params = profile(model, (fus, events ))
    print('flops: ', flops, 'params: ', params)#直接print输出的是个数
    # 转换后的M表示百万个，不是存储单位，G表示每秒10亿 (=10^9) 次的浮点运算，
    print('》》》》》》》》》》》Flops: ', str(flops / 1000 ** 3) + 'G')
    print('》》》》》》》》》》》Params: ', str(params / 1000 ** 2) + 'M')
