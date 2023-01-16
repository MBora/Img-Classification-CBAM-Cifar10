import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2


# CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
batch_size = 10
num_labels = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# For train set
train_transformations = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # randomly crop the image
    transforms.RandomHorizontalFlip(),  # randomly flip the image horizontally
    # randomly jitter the image colors
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),  # convert the image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                         )  # normalize the image
])

train_set = CIFAR10(root='./data', train=True,
                    transform=train_transformations, download=True)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0)

#For test set
test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Apply the transformations to the test set
test_set = CIFAR10(root='./data', train=False,
                   transform=test_transformations, download=True)

# Create a loader for the transformed test set
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=0)

# Print the number of images in the training set
print("No. of images in one test set is: ", len(test_loader)*batch_size)
print("No. of batches per epoch is: ", len(train_loader))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels//16, out_channels=in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        spatial_attention = self.sigmoid(x)
        return spatial_attention

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels//16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels//16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channels//16, out_channels=channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        spatial_attention = self.sigmoid(x)
        return spatial_attention

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention(channels)

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)
        attention = channel_attention * spatial_attention
        out = attention * x
        return x * attention, channel_attention, spatial_attention
        


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)
        self.cbam = CBAM(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 10)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x, channel_attention, spatial_attention = self.cbam(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        return x, channel_attention, spatial_attention


model = CNN()

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_func = nn.CrossEntropyLoss()


def saveModel():
    path = "./myFinalModel.pth"
    torch.save(model.state_dict(), path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The model will run on", device)
model.to(device)

# validation
def validate():
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for data in test_loader:
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            outputs, channel, spatial = model(image)

            # loss = F.cross_entropy(outputs, labels)
            loss = loss_func(outputs, labels)
            valid_running_loss += loss.item()

            # calculate the accuracy
            _, predicted = torch.max(outputs.data, 1)
            valid_running_correct += (predicted == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    validation_loss = valid_running_loss / counter
    validation_acc = 100. * (valid_running_correct / len(test_loader.dataset))
    return validation_loss, validation_acc


def train(num_epochs):
    
    best_accuracy = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs, _, _ = model(images)

            # loss = F.cross_entropy(outputs, labels)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:    
                print('[epoch:%d, minibatch:%5d] loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
        #validation
        validation_loss, validation_acc = validate()
        print('Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(validation_loss, validation_acc))

        # save the model if the validation accuracy is the best we've seen so far.
        #for cifar10 it is ok to save model with best accuracy
        
        if validation_acc > best_accuracy:
            print('Validation accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_accuracy, validation_acc))
            saveModel()
            best_accuracy = validation_acc

        #we can also save model with best loss
        #uncomment this to save model with best loss
        # if validation_loss < best_loss:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_loss, validation_loss))
        #     saveModel()
        #     best_loss = validation_loss

def AccuracyPerClass(model, test_loader):
    model.eval()
    correct = list(0. for i in range(num_labels)) 
    total = list(0. for i in range(num_labels)) 
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs, _, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                correct[label] += c[i].item()
                total[label] += 1

    for i in range(num_labels):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * correct[i] / total[i])) 


if __name__ == "__main__":
    
    # Let's build our model
    #after 8 epochs model seems to be overfitting
    train(10)
    print('Finished Training')
    img = Image.open("gata_cat_cat_face.jpg")
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # img.show()
    img = transform(img).unsqueeze(0)


    model = CNN()
    path = "myFinalModel.pth"
    model.load_state_dict(torch.load(path))
    outputs, channel_attention_map, spatial_attention_map = model(img)
    AccuracyPerClass(model, test_loader)


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(channel_attention_map[0, 0].detach().numpy(), cmap='jet')
    ax1.set_title("channel attention map")
    ax2.imshow(spatial_attention_map[0, 0].detach().numpy(), cmap='jet')
    ax2.set_title("spatial attention map")
    ax3.imshow(outputs.detach().numpy(), cmap='jet')
    ax3.set_title("output")
    plt.show()
