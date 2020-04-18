import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from MNIST_model import *

def data_load():

    # MNIST dataset 다운로드
    train_data = dsets.MNIST(root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
    val_data = dsets.MNIST(root="./dataset/", train=False, transform=transforms.ToTensor(), download=True)

    return train_data, val_data


def imgshow(image, label):
    print('========================================')
    print("The 1st image:")
    print(image)
    print('Shape of this image\t:', image.shape)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title('Label:%d' % label)
    plt.show()
    print('Label of this image:', label)


def generate_batch(train_data, val_data):
    train_batch_loader = DataLoader(train_data, cfg.batch_size, shuffle=True)
    val_batch_loader = DataLoader(val_data, cfg.batch_size, shuffle=True)
    return train_batch_loader, val_batch_loader


if __name__ == '__main__':
    print('[MNIST_training]')
    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # configuration
    cfg = Config()

    # 데이터 로드
    # MNIST datset: 28 * 28 사이즈의 이미지들을 가진 dataset
    train_data, val_data = data_load()

    # data 개수 확인
    print('The number of training data: ', len(train_data))
    print('The number of validation data: ', len(val_data))

    # shape 및 실제 데이터 확인
    image, label = train_data[0]
    imgshow(image, label)

    # 학습 모델 생성
    model = MNIST_model().to(device)

    # 배치 생성
    train_batch_loader, val_batch_loader = generate_batch(train_data, val_data)

    ###############################################################################################################
    #                  TODO : 모델 학습을 위한 optimizer 정의                                                       #
    ###############################################################################################################
    pass
    ###############################################################################################################
    #                                              END OF YOUR CODE                                               #
    ###############################################################################################################

    criterion = nn.CrossEntropyLoss()

    # training 시작
    start_time = time.time()
    highest_val_acc = 0
    val_acc_list = []
    print('========================================')
    print("Start training")
    for epoch in range(cfg.epoch):
        train_loss = 0
        train_batch_cnt = 0
        model.train()
        for img, label in train_batch_loader:
            # img.shape: [200,1,28,28]
            # label.shape: [200]
            img = img.to(device)
            label = label.to(device)

            # input data shape: [200,28*28]

            ##############################################################################################################
            #              TODO : foward path를 진행하고 손실을 loss에 저장 후 train_loss에 더함, 모델 학습 진행              #
            ##############################################################################################################
            pass
            ###############################################################################################################
            #                                              END OF YOUR CODE                                               #
            ###############################################################################################################

            train_batch_cnt += 1
        ave_loss = train_loss / train_batch_cnt
        training_time = (time.time() - start_time) / 60
        print('========================================')
        print("epoch:", epoch + 1)
        print("training dataset average loss: %.3f" % ave_loss)
        print("training_time: %.2f minutes" % training_time)

        # validation (for early stopping)
        correct_cnt = 0
        model.eval()
        for img, label in val_batch_loader:
            img = img.to(device)
            label = label.to(device)
            pred = model.forward(img.view(-1, 28 * 28))
            _, top_pred = torch.topk(pred, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            correct_cnt += int(torch.sum(top_pred == label))

        val_acc = correct_cnt / len(val_data) * 100
        print("validation dataset accuracy: %.2f" % val_acc)
        val_acc_list.append(val_acc)
        if val_acc > highest_val_acc:
            save_path = './saved_model/setting_1/epoch_' + str(epoch + 1) + '.pth'
            # 위와 같이 저장 위치를 바꾸어 가며 각 setting의 epoch마다의 state를 저장할 것.
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)
            highest_val_acc = val_acc

    print("Training finished.")

    epoch_list = [i for i in range(1, 101)]
    plt.title('Validation dataset accuracy plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epoch_list, val_acc_list)
    plt.show()