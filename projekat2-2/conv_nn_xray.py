import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from visualisation import create_acc_loss_graph

REBUILD_DATA = False # set to true to one once, then back to false unless you want to change something in your training data.
REBUILD_TEST_DATA = False

device = torch.device("cuda:0")

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


class XRAYClassification():
    IMG_SIZE = 100
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
        self.conv1 = nn.Conv2d(1, 8, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(8, 16, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(16, 32, 5)
        #self.conv4 = nn.Conv2d(32, 64, 5)
        #self.conv5 = nn.Conv2d(64, 128, 5)
        #self.conv6 = nn.Conv2d(512, 1024, 3)
        #self.conv7 = nn.Conv2d(1024, 2048, 3)
        #self.conv8 = nn.Conv2d(2048, 4096, 2)
        #self.conv9 = nn.Conv2d(4096, 8192, 2)

        x = torch.randn(100,100).view(-1,1,100,100)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 3)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.leaky_relu(self.conv3(x)), (2,2))
        #x = F.leaky_relu(self.conv4(x))
        #x = F.leaky_relu(self.conv5(x))
        #x = F.relu(self.conv6(x))
        #x = F.relu(self.conv7(x))
        #x = F.relu(self.conv8(x))
        #x = F.relu(self.conv9(x))


        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)



if REBUILD_DATA:
    xray = XRAYClassification()
    xray.make_training_data()

if REBUILD_TEST_DATA:
    xray = XRAYClassification()
    xray.make_test_data()


net = Net()
net.to(device)
print(net)

training_data = np.load("training_data.npy", allow_pickle=True)
test_data = np.load("test_data.npy", allow_pickle=True)
#print(len(training_data))

optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,100,100)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)

val_X = torch.Tensor([i[0] for i in test_data]).view(-1,100,100)
val_X = val_X/255.0
val_y = torch.Tensor([i[1] for i in test_data])

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]




BATCH_SIZE = 100

MODEL_NAME = f"model-{int(time.time())}"

def train(net):
    BATCH_SIZE = 100
    EPOCHS = 30

    with open("model.log", "w") as f:
        for epoch in range(EPOCHS):
            print("\nCurrent epoch: ", epoch)

            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 100, 100)
                batch_y = train_y[i:i + BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                # print(f"Acc: {round(float(acc),2)}  Loss: {round(float(loss),4)}")
                # f.write(f"{MODEL_NAME},{round(time.time(),3)},train,{round(float(acc),2)},{round(float(loss),4)}\n")
                # just to show the above working, and then get out:
                if i % 50 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(
                        f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")

def test(size=32):
    random_start = np.random.randint(len(test_X)-size)
    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1, 1, 100, 100).to(device), y.to(device))
    return val_acc, val_loss


def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss

def final_test(net):
    correct = 0
    total = 0
    for i in tqdm(range(0, len(test_X), BATCH_SIZE)):

        batch_X = test_X[i:i + BATCH_SIZE].view(-1, 1, 100, 100).to(device)
        batch_y = test_y[i:i + BATCH_SIZE].to(device)
        batch_out = net(batch_X)

        out_maxes = [torch.argmax(i) for i in batch_out]
        target_maxes = [torch.argmax(i) for i in batch_y]
        for i, j in zip(out_maxes, target_maxes):
            if i == j:
                correct += 1
            total += 1
    print("\nAccuracy: ", round(correct / total, 3))

train(net)
final_test(net)
create_acc_loss_graph(MODEL_NAME)




