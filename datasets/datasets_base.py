# * datasets_base.py:
#   base class template of "Trainer"
#
# * Test Status: Test OK
#

# -*- coding: utf-8 -*
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

class DatasetsBase(data.Dataset):
    '''
    DatasetsBase: Write self 'Dataset' class easily by inheriting this class
    '''

    def __init__(self, dataset_filepath):
        self.filepath = dataset_filepath

        self.data = []
        self.labels = []

        self.transform = transforms.Compose([])

        self.pattern = []
        self.target_w = 0
        self.target_h = 0


    def setup(self):
        '''
        - When overriding: Read the file of the datasets, and load into the object
        - Alarm: Please call 'super' method at the end of the override method
        '''
        if not isinstance(self.data, list):
            raise TypeError("DatasetsBase: 'self.data' should be 'list'.")
        if not isinstance(self.labels, list):
            raise TypeError("DatasetsBase: 'self.labels' should be 'list'.")

        if not isinstance(self.transform, transforms.Compose):
            raise TypeError("DatasetsBase: 'self.train_transform''s values must be as type 'transforms.Compose'.")

        if not isinstance(self.pattern, list):
            raise TypeError("DatasetsBase: 'self.pattern' should be 'list'.")

        if not isinstance(self.target_w, int) or not isinstance(self.target_h, int):
            raise TypeError("DatasetsBase: 'self.target_w' & 'self.target_h' should be 'int'.")

        if len(self.data)*len(self.labels)==0:
            raise ValueError("DatasetsBase: variable(s) below may be null:"
                             "'self.data' is {}, "
                             "'self.labels' is {}.".format(self.data,self.labels))


    def getinfo(self):
        return self.pattern, self.target_w, self.target_h


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        '''
        - builtin method '__getitem__' for getting item of DatasetsBase object by sending index
        - P.S. Can be overrided without executing 'super' method
        :param idx: index of the item which needs to get
        :return: img: PIL Image Object; labels: Label
        '''
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]

