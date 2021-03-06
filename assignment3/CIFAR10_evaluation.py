import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CIFAR10_configuration import Config
from LeNet5_model import LeNet5_model
from MLP_model import MLP_model
from ResNet_model import ResNet32_model
def data_load():

    # CIFAR10 dataset 다운로드
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_data = dsets.CIFAR10(root='./dataset/', train=False, transform=transforms_test, download=True)
    return test_data


def generate_batch(test_data):
    test_batch_loader = DataLoader(test_data, cfg.batch_size, shuffle=True)
    return test_batch_loader


if __name__ == "__main__":
    print('[CIFAR10_evaluation]')
    cfg = Config()

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # GPU 사용시
    if torch.cuda.is_available():
        torch.cuda.device(0)

    # 모델 생성
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
    save_path = "./saved_model/setting_3/epoch_145.pth"
    # TODO : 세팅값 마다 save_path를 바꾸어 로드
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    correct_cnt = 0
    for img, label in test_batch_loader:
        img = img.to(device)
        label = label.to(device)
        pred = model.forward(img)
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)

        correct_cnt += int(torch.sum(top_pred == label))

    accuracy = correct_cnt / len(test_data) * 100
    print("accuracy of the 35 epoch trained model:%.2f%%" % accuracy)
    acc_list.append(accuracy)


