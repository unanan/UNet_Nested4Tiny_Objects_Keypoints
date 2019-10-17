import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Resnet_m(nn.Module):
    ''' RNet '''

    def __init__(self, block = Bottleneck, num_blocks = [2,3,4,2],is_train=False, use_cuda=True):
        super(Resnet_m, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        #region original Onet Pre Layer
        # backend
        # self.pre_layer = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
        #     nn.PReLU(),  # prelu1
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
        #     nn.PReLU(),  # prelu2
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
        #     nn.PReLU(), # prelu3
        #     nn.MaxPool2d(kernel_size=2,stride=2), # pool3
        #     nn.Conv2d(64,128,kernel_size=2, stride=1), # conv4
        #     nn.PReLU(), # prelu4
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # pool4
        #     nn.Conv2d(128, 256, kernel_size=2, stride=1),  # conv5
        #     nn.PReLU(),  # prelu5
        #     nn.MaxPool2d(kernel_size=1, stride=1),  # pool5
        #     nn.Conv2d(256, 256, kernel_size=2, stride=1),  # conv5
        #     nn.PReLU()  # prelu6
        # )
        #endregion

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, 10, kernel_size=3, stride=1, padding=1)

        self.linklayer = nn.Linear(49152, 10)


        # self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # conv5
        # self.conv5_1 = nn.Linear(16384, 1024)  # conv5
        #
        # self.conv5_2 = nn.Linear(1024, 256)  # 128*2*2
        # self.prelu5 = nn.PReLU()  # prelu5
        # # detection
        # self.conv6_1 = nn.Linear(256, 1)
        # # bounding box regression
        # self.conv6_2 = nn.Linear(256, 4)
        # # lanbmark localization
        # self.conv6_3 = nn.Linear(256, 10)  #10
        # weight initiation weih xavier
        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # backend
        # x = self.pre_layer(x)
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        x = x.view(x.size(0), -1)
        landmark = torch.sigmoid(self.linklayer(x)) *10.0

        # p1 = self._upsample_add(p2, self.latlayer4(c1))
        # landmark = self.toplayer3(p1)

        # c5 = self.downlayer1(c4)
        # x = self.downlayer2(x)
        # x = self.layer4(x)
        
        # x = x.view(x.size(0), -1)

        # x = self.conv5_1(x)
        # x = self.conv5_2(x)
        # x = self.prelu5(x)
        # landmark = torch.sigmoid(self.conv6_3(x))
        # if self.is_train is True:
        #     return  landmark
        return landmark


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)

class ONet_1(nn.Module):

    def __init__(self,is_train=False, use_cuda=True):

        super(ONet_1, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),
            ('pool3', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv5', nn.Conv2d(128, 128, 2, 1)),
            ('prelu5', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv6', nn.Linear(18432, 1024)),
            ('drop6', nn.Dropout(0.25)),
            ('prelu6', nn.PReLU(1024)),

            ('conv7', nn.Linear(1024, 256)),
            ('drop7', nn.Dropout(0.25)),
            ('prelu7', nn.PReLU(256)),
            #
            # ('conv7', nn.Linear(1024, 256)),
            # ('drop7', nn.Dropout(0.25)),
            # ('prelu7', nn.PReLU(256)),
        ]))

        # self.conv6_1 = nn.Linear(256, 2)
        # self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)
        self.apply(weights_init)
        # weights = np.load('src/weights/onet.npy')[()]
        # for n, p in self.named_parameters():
        #     p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        # a = self.conv6_1(x)
        # b = self.conv6_2(x)
        x = self.conv6_3(x)
        landmark = torch.sigmoid(x) *4.0

        # a = F.softmax(a)
        return landmark



