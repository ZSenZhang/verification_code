"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
torchvision
matplotlib
"""
# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pretreatment

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 6               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
# root='./mnist/raw/'

def changepicsize(path,maxheight):
    fh = open(path+'.txt','r')
    f=open('new_new_'+path+'.txt','w')
    i=-1
    for line in fh:
        line=line.strip('\n')
        line = line.rstrip()
        words = line.split()
        if(words[0]=='.'):
            continue
        if(words[0]=='..'):
            continue
        img2=Image.new('RGB',(32,32),(0,0,0))
        img = Image.open('new_'+path+'/'+words[0])
        
        box1=(0,0,32,32)
        region=img.crop(box1)
        img2.paste(region,(0,0))
        if(not os.path.exists('new_new_'+path+'/')):
            os.makedirs('new_new_'+path+'/')
        img2.save('new_new_'+path+'/'+words[0])
        if(len(words)>1):
            f.write('new_new_'+path+'/'+words[0]+' '+words[1]+'\n')
        else:
            f.write('new_new_'+path+'/'+words[0]+' '+words[0][0]+'\n')
        # i=i+1


def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self,txt,transform=None,target_transform=None,loader=default_loader):
        fh=open(txt,'r')
        imgs=[]
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs=imgs
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader

    def __getitem__(self,index):
        fn,label = self.imgs[index]
        img=self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

changepicsize('train',32)
#changepicsize('test',32)
train_data=MyDataset(txt='./new_new_train.txt',transform=torchvision.transforms.ToTensor())
test_data=MyDataset(txt='./new_new_test.txt',transform=torchvision.transforms.ToTensor())

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(dataset=test_data,batch_size=64)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 32, 32)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 16384, 16384)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 8192, 8192)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 16, 16)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 16, 16)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 8, 8)
        )
        self.out = nn.Linear(32 * 8 * 8, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
#  !!!!add cuda()!!!
# cnn.cuda()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('Please install sklearn for layer visualization')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

# plt.ion()
# training and testing
train_acc_all=[]
size_all=[]
count_size=0
for epoch in range(EPOCH):
    train_loss = 0.
    train_acc = 0.
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
    # !!! add cuda()?
        b_x = Variable(x)   # batch x
        b_y = Variable(y)  # batch y

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        train_loss+=loss.data[0]
        pred=torch.max(output,1)[1].data.squeeze()
        train_acc=sum(pred==b_y).item()/float(b_y.size(0))
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        train_acc_all.append(train_acc)
        size_all.append(count_size)
        count_size=count_size+1

    # print()

        if step % 50 == 0:
            # test_output, last_layer = cnn(test_x)
            # pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # accuracy = sum(pred_y == test_y).item() / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % (train_loss / (len(train_data))), '| test accuracy: %.2f' % (train_acc),'| pred_y:',pred[:10],'|train_y:',b_y[:10].numpy())
            # if HAS_SK:
            #     # Visualization of trained flatten layer (T-SNE)
            #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            #     plot_only = 500
            #     low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            #     labels = test_y.numpy()[:plot_only]
            #     plot_with_labels(low_dim_embs, labels)
# plt.ioff()
torch.save(cnn,'cnn.pkl')
plt.plot(size_all,train_acc_all)
plt.show()
# print 10 predictions from test data
cnn.eval()
eval_loss = 0.
eval_acc = 0.
for batch_x, batch_y in test_loader:
    #!!!!add cuda()!!!!
    batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
    out, _ = cnn(batch_x[:10])
    # outputx = cnn(batch_x)[0]
    # loss = loss_func(outputx, batch_y)
    # eval_loss += loss.data[0]
    #!!!!add cuda()!!!!!
    pred = torch.max(out, 1)[1].data.numpy().squeeze()
     num_correct = sum(pred == batch_y).item()/float(batch_y.size(0))
     eval_acc += num_correct.data[0]
# pred_y=pred.data.numpy().squeeze()
print(pred,'predictions number')
print(batch_y[:10].numpy(),'real number')
 print('Acc:%.2f',num_correct)



