import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False # set to true to one once, then back to false unless you want to change something in your training data.
REBUILD_TEST_DATA = False

class XRAYClassification():
    IMG_SIZE = 50
    DATA = "chest_xray_data_set"
    TESTDATA = "chest-xray-dataset-test" + os.path.sep + "test"
    NORMAL = "normal"
    VIRUS = "virus"
    BACTERIA = "bacteria"
    LABELS = {NORMAL: 0, VIRUS: 1, BACTERIA: 2}
    training_data = []
    test_data = []

    viruscount = 0
    normalcount = 0
    bacteriacount = 0

    def make_training_data(self):
        f = open("metadata" + os.path.sep + "chest_xray_metadata.csv","r")
        lines = f.readlines()
        lines = lines[1:]

        data = []

        for line in lines:
            pom = line.split(",")
            data.append([pom[1], pom[-1].replace("\n","")])

        #print(data.__contains__(["IM-0128-0001.jpeg", ""]))
        for f in tqdm(os.listdir(self.DATA)):
            if "jpeg" or "jpg" or "png" in f:
                try:
                    path = os.path.join(self.DATA, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    label = ""
                    if data.__contains__([f, ""]):
                        label = "normal"
                    elif data.__contains__([f, "Virus"]):
                        label = "virus"
                    elif data.__contains__([f, "bacteria"]):
                        label = "bacteria"
                    else:
                        continue

                    if label == self.NORMAL:
                        self.normalcount += 1
                    elif label == self.VIRUS:
                        self.viruscount += 1
                    else:
                        if self.bacteriacount < 1600:
                            self.bacteriacount += 1
                        else:
                            continue
                    self.training_data.append([np.array(img), np.eye(3)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot
                    #print(np.eye(2)[self.LABELS[label]])
                except Exception as e:
                    pass
                    #print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Bactery:',self.bacteriacount)
        print('Normal:',self.normalcount)
        print('Virus:',self.viruscount)

    def make_test_data(self):
        f = open("chest-xray-dataset-test" + os.path.sep + "chest_xray_test_dataset.csv","r")
        lines = f.readlines()
        lines = lines[1:]

        data = []

        for line in lines:
            pom = line.split(",")
            data.append([pom[1], pom[-1].replace("\n","")])

        #print(data.__contains__(["IM-0128-0001.jpeg", ""]))
        for f in tqdm(os.listdir(self.TESTDATA)):
            if "jpeg" or "jpg" or "png" in f:
                try:
                    path = os.path.join(self.TESTDATA, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    label = ""
                    if data.__contains__([f, ""]):
                        label = "normal"
                    elif data.__contains__([f, "Virus"]):
                        label = "virus"
                    elif data.__contains__([f, "bacteria"]):
                        label = "bacteria"
                    else:
                        continue

                    if label == self.NORMAL:
                        self.normalcount += 1
                    elif label == self.VIRUS:
                        self.viruscount += 1
                    else:
                        self.bacteriacount += 1
                    self.test_data.append([np.array(img), np.eye(3)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot
                    #print(np.eye(2)[self.LABELS[label]])
                except Exception as e:
                    pass
                    #print(label, f, str(e))

        np.random.shuffle(self.test_data)
        np.save("test_data.npy", self.test_data)
        print('Bactery:',self.bacteriacount)
        print('Normal:',self.normalcount)
        print('Virus:',self.viruscount)


class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 3) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


net = Net()
#print(net)

if REBUILD_DATA:
    xray = XRAYClassification()
    xray.make_training_data()

if REBUILD_TEST_DATA:
    xray = XRAYClassification()
    xray.make_test_data()

training_data = np.load("training_data.npy", allow_pickle=True)
test_data = np.load("test_data.npy", allow_pickle=True)
#print(len(training_data))

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

#VAL_PCT = 0.1  # lets reserve 10% of our data for validation
#val_size = int(len(X))

val_X = torch.Tensor([i[0] for i in test_data]).view(-1,50,50)
val_X = val_X/255.0
val_y = torch.Tensor([i[1] for i in test_data])

train_X = X
train_y = y

test_X = val_X #izmena
test_y = val_y #izmena

BATCH_SIZE = 100
EPOCHS = 2



def train(net):
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            #print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i+BATCH_SIZE]

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()    # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")


def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list,
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct/total, 3))

train(net)
test(net)



