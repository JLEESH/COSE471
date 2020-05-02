import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from CIFAR10_configuration import Config
from LeNet5_model import LeNet5_model
from MLP_model import MLP_model
from ResNet_model import ResNet32_model
from torch.optim import lr_scheduler


def data_load():
    # train data augmentation : 1) 데이터 좌우반전(2배). 2) size 4만큼 패딩 후 32의 크기로 random cropping
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR10 dataset 다운로드
    train_data = dsets.CIFAR10(root='./dataset/', train=True, transform=transforms_train, download=True)
    val_data = dsets.CIFAR10(root="./dataset/", train=False, transform=transforms_val, download=True)

    return train_data, val_data


def imgshow(image, label, classes):
    print('========================================')
    print("The 1st image:")
    print(image)
    print('Shape of this image\t:', image.shape)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.title('Label:%s' % classes[label])
    plt.show()
    print('Label of this image:', label, classes[label])


def generate_batch(train_data, val_data):
    train_batch_loader = DataLoader(train_data, cfg.batch_size, shuffle=True)
    val_batch_loader = DataLoader(val_data, cfg.batch_size, shuffle=True)
    return train_batch_loader, val_batch_loader


if __name__ == '__main__':
    # configuration
    cfg = Config()

    print('[CIFAR10_training]')
    print('Training with:', cfg.modelname)
    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # GPU 사용시
    # if torch.cuda.is_available():
    #     torch.cuda.device("cuda:2")

    # 데이터 로드
    # CIFAR10 dataset: [3,32,32] 사이즈의 이미지들을 가진 dataset
    train_data, val_data = data_load()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # data 개수 확인
    print('The number of training data: ', len(train_data))
    print('The number of validation data: ', len(val_data))

    # shape 및 실제 데이터 확인
    image, label = train_data[1]
    imgshow(image, label, classes)

    # 학습 모델 생성

    if cfg.modelname == "MLP":
        model = MLP_model()
    elif cfg.modelname == "LeNet5":
        model = LeNet5_model()
    elif cfg.modelname == "ResNet32":
        model = ResNet32_model()
    else:
        print("Wrong modelname.")
        quit()
    if torch.cuda.is_available():
        model = model.to(device)

    # 배치 생성
    train_batch_loader, val_batch_loader = generate_batch(train_data, val_data)
    criterion = nn.CrossEntropyLoss()

    if cfg.modelname == "MLP":
        ###############################################################################################################
        #                  TODO : 모델 학습을 위한 optimizer 정의                                                       #
        ###############################################################################################################
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        decay_epoch = [32000, 48000]
        step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

    elif cfg.modelname == "LeNet5":
        ###############################################################################################################
        #                  TODO : 모델 학습을 위한 optimizer 정의                                                       #
        ###############################################################################################################
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        decay_epoch = [32000, 48000]
        step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################
    elif cfg.modelname == "ResNet32":
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        decay_epoch = [32000, 48000]
        step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=decay_epoch, gamma=0.1)

    # training 시작
    start_time = time.time()
    highest_val_acc = 0
    val_acc_list = []
    global_steps = 0
    epoch = 0
    print('========================================')
    print("Start training...")
    while True:
        train_loss = 0
        train_batch_cnt = 0
        model.train()
        for img, label in train_batch_loader:
            global_steps += 1
            # img.shape: [200,3,32,32]
            # label.shape: [200]

            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss

            train_batch_cnt += 1
            #quit()
            if global_steps >= 64000:
                print("Training finished.")
                break

        ave_loss = train_loss / train_batch_cnt
        training_time = (time.time() - start_time) / 60
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % ave_loss)
        print("training_time: %.2f minutes" % training_time)

        # validation (for early stopping)
        correct_cnt = 0
        model.eval()
        for img, label in val_batch_loader:
            img = img.to(device)
            label = label.to(device)
            pred = model.forward(img)
            _, top_pred = torch.topk(pred, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            correct_cnt += int(torch.sum(top_pred == label))

        val_acc = correct_cnt / len(val_data) * 100
        print("validation dataset accuracy: %.2f" % val_acc)
        val_acc_list.append(val_acc)
        if val_acc > highest_val_acc:
            save_path = './saved_model/setting_3/epoch_' + str(epoch + 1) + '.pth'
            # 위와 같이 저장 위치를 바꾸어 가며 각 setting의 epoch마다의 state를 저장할 것.
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)
            highest_val_acc = val_acc
        epoch += 1
        if global_steps >= cfg.finish_step:
            break

    epoch_list = [i for i in range(1, epoch + 1)]
    plt.title('Validation dataset accuracy plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epoch_list, val_acc_list)
    plt.show()
