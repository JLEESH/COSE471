class Config():
    def __init__(self):
        self.modelname = "ResNet32"  # MLP / LeNet5 / ResNet32
        self.batch_size = 128
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-04
        self.finish_step = 64000
