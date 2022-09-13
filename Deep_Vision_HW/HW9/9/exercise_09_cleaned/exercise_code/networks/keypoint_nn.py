"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim as op
import torch.nn.functional as F

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams, train_data):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams.update(hparams)
        self.train_set = train_data
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################

        self.conv = nn.Sequential(
                #layer 1
                nn.Conv2d(1, 64, kernel_size=16, stride=1), #64*81*81
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #64*40*40
                #layer 2
                nn.Conv2d(64, 128, kernel_size=16, stride=1), #128*25*25
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #128*12*12
                #layer 3
                nn.Conv2d(128, 256, kernel_size=5, stride=1), #256*8*8
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2), #256*3*3
            )
            
        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2304, 512), 
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(64, self.hparams["output"])
            )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        
        x = self.conv(x)
        x = self.fc(x)

        return x
        
    def training_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["keypoints"]
        
        out = self.forward(images).resize(15,2)
        loss = F.mse_loss(out, targets)

        self.log('loss', loss) 
        tensorboard_logs = {'loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'])
        
      
    def configure_optimizers(self):

        optim = op.Adam(self.parameters(), lr=self.hparams['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience = 10)
        
        return {
          'optimizer': optim,
          'lr_scheduler': scheduler, 
          'monitor': 'loss'
        }
 
        
        
      

        
        
class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
