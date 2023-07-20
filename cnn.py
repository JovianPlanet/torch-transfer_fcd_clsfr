import torch
from torch import nn


class Cnn2D(nn.Module):

    def __init__(self, in_channels, classes, start_filters=16):
        # call the parent constructor
        super(Cnn2D, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=start_filters, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=start_filters, out_channels=start_filters, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # initialize third set of CONV => RELU => POOL layers
        self.conv3 = nn.Conv2d(in_channels=start_filters, out_channels=start_filters*2, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # initialize fourth set of CONV => RELU => POOL layers
        self.conv4 = nn.Conv2d(in_channels=start_filters*2, out_channels=start_filters*4, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=64*6*6)
        self.relu5 = nn.ReLU()

        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=64*6*6, out_features=classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input => CONV => RELU => POOL layers
        # print(f'Input dims = {x.shape}')
        x = self.conv1(x)
        # print(f'1era conv = {x.shape}')
        x = self.relu1(x)
        x = self.maxpool1(x)
        # print(f'1er maxpool = {x.shape}')

        x = self.conv2(x)
        # print(f'2da conv = {x.shape}')
        x = self.relu2(x)
        x = self.maxpool2(x)
        # print(f'2do maxpool = {x.shape}')

        x = self.conv3(x)
        #print(f'3era conv = {x.shape}')
        x = self.relu3(x)
        x = self.maxpool3(x)
        #print(f'3er maxpool = {x.shape}')

        x = self.conv4(x)
        #print(f'4ta conv = {x.shape}')
        x = self.relu4(x)
        x = self.maxpool4(x)
        #print(f'4t maxpool = {x.shape}')

        # flatten the output from the previous layer and pass it through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # pass the output to our softmax classifier to get our output predictions
        x = self.fc2(x)
        output = self.sigmoid(x)
        # return the output predictions
        
        return output