import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image

batch_size = 10
num_labels = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# transformations
#Apply transformations to train set
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
#DataLoader To convert data into
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0)


test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Apply the transformations to the test set
test_set = CIFAR10(root='./data', train=False,
                   transform=test_transformations, download=True)

test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=0)

print("No. of batches per epoch is: ", len(train_loader))

# To compute Channel attention
class ChannelAttentionModule(nn.Module): 
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=3, padding=1, bias=False) #kernel size and padding can vary according to requirement
        self.relu = nn.ReLU() #Activation function between convolution layers
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_module = self.fc2(self.relu(self.fc1(self.avg_pool(x)))) #average pool channel
        max_pool_module = self.fc2(self.relu(self.fc1(self.max_pool(x)))) #max pool channel
        
        out = self.sigmoid(avg_pool_module + max_pool_module) #out is the attention map
        refined_feature_map = out * x #out*x is the refined feature map or the channel attention map
        return refined_feature_map

#To compute spatial attention
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        #input channels come from one average channel and one spatial channel
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1) #kernel size and padding can vary according to requirement
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = torch.mean(x, dim=1, keepdim=True) #Average pooling along a dimension
        max_x, _ = torch.max(x, dim=1, keepdim=True) #Max pooling along a dimension
        new_x = torch.cat([avg_x, max_x], dim=1) #Concatenation of both tesnors
        #Layer inside spatial module
        new_x = self.conv1(new_x)
        new_x = self.relu1(new_x)
        new_x = self.conv2(new_x)
        new_x = self.sigmoid(new_x)
        refined_feature_map = new_x * x #new_x*x is the refined feature map or the spatial attention map; x was the channel attention
        return refined_feature_map

#To combine channel and spatial attention modules i.e implement CBAM module
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttentionModule(channels)
        self.sa = SpatialAttentionModule()

    def forward(self, x):
        x = self.ca(x)
        channel_attention = x.clone().detach()
        x = self.sa(x)
        spatial_attention = x.clone().detach()
        return x, channel_attention, spatial_attention #spatial_attention and x is the same

#Neural Network
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
        self.dropout1 = nn.Dropout(p=0.25) #0.25 gave better performance than p=0.5
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
    path = "./FinalModel.pth"
    torch.save(model.state_dict(), path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The model will run on", device)
model.to(device)

# validation function to calculate the accuracy and loss for the validation set
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

            outputs, _, _ = model(image)

            # loss = F.cross_entropy(outputs, labels) #either works
            loss = loss_func(outputs, labels)
            valid_running_loss += loss.item()

            # calculate the accuracy
            _, predicted = torch.max(outputs.data, 1)
            valid_running_correct += (predicted == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    validation_loss = valid_running_loss / counter
    validation_acc = 100. * (valid_running_correct / len(test_loader.dataset))
    return validation_loss, validation_acc

# training
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
        #uncomment this to save model with best loss by also commenting validation accuracy section
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

    print('Starting Training')
    #Overfitting after 15-18 epochs
    # train(8)
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
    path = "FinalModel.pth"
    model.load_state_dict(torch.load(f"{path}", map_location=torch.device(device=device)))
    
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