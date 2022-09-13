"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn.functional as F
from torch import optim as op
from torch.autograd import Variable
import torchvision.models.segmentation as models

class SegmentationNN(nn.Module):
    def __init__(self,num_classes=23):
        super(SegmentationNN, self).__init__()
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.conv = nn.Sequential(
                #layer 1
                nn.Conv2d(3, 32, kernel_size=32, stride=1), 
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), 
                #layer 2
                nn.Conv2d(32, 64, kernel_size=16, stride=1), 
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), 
                #layer 3
                nn.Conv2d(64, 256, kernel_size=7, stride=1), 
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.fcn = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2), 
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(256, 2048, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(2048, num_classes, kernel_size=1) 
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        out = self.conv(x)
        out = self.fcn(out)
        out = F.interpolate(out, size=x.size()[2:])
        

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return out

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
