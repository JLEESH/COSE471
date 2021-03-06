import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from MNIST_model import *


def data_load():

    # MNIST dataset 다운로드
    test_data = dsets.MNIST(root='./dataset/', train=False, transform=transforms.ToTensor(), download=True)
    return test_data


def generate_batch(test_data):
    test_batch_loader = DataLoader(test_data, cfg.batch_size, shuffle=True)
    return test_batch_loader


if __name__ == "__main__":
    print('[MNIST_evaluation]')
    cfg = Config()

    # 모델 생성
    model = MNIST_model()
    model.eval()

    # 데이터 로드
    test_data = data_load()

    # data 개수 확인
    print('The number of test data: ', len(test_data))

    # 배치 생성
    test_batch_loader = generate_batch(test_data)

    # test 시작
    acc_list = []


    # 저장된 state 불러오기
    # added setting and best_epoch to evaluate models

    import argparse
    from argparse import RawTextHelpFormatter
    desc = 'COSE471 Assignment 2'\
        + '\nInput settings'\
        + '\n\ne.g. python(3) MNIST_evaluation.py -s 4 -e 100 \n'
    parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-s', '--setting', default=4, type=int, choices=[1, 2, 3, 4], help='Setting to use (default: 4).')
    parser.add_argument('-e', '--epoch', default=1, type=int, help='Epoch number (default: 1).')
    args = parser.parse_args()
    print('Setting: ' + str(args.setting))
    print('Epoch: ' + str(args.epoch))

    save_path = "./saved_model/setting_" + str(args.setting) + "/epoch_" + str(args.epoch) + ".pth"
    # TODO : 세팅값 마다 save_path를 바꾸어 로드
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    correct_cnt = 0
    for img, label in test_batch_loader:
        pred = model.forward(img.view(-1, 28 * 28))
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)

        correct_cnt += int(torch.sum(top_pred == label))

    accuracy = correct_cnt / len(test_data) * 100
    print("accuracy of the 87 epoch trained model:%.2f%%" % accuracy)
    acc_list.append(accuracy)


