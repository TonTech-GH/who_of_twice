import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from twice_cnn import TwiceCNN


class TwiceClassifier:
    def __init__(self, image_root, batch_size=32):
        self.image_root = image_root

        # GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device:', self.device)

        # ランダムシードの固定
        pl.seed_everything(0)

        # バッチサイズ
        self.batch_size = batch_size

        # 前処理の定義
        self.valid_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        self.train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),  # ランダムに左右反転
            transforms.ColorJitter(),  # ランダムに画像の色調を変更
            transforms.RandomRotation(10),  # ランダムに画像回転(±10度)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def load_images(self, image_root):
        # 画像の読み込み
        TRAIN_ROOT = f'{IMAGE_ROOT}/train'
        VALID_ROOT = f'{IMAGE_ROOT}/valid'

        train_dataset = datasets.ImageFolder(TRAIN_ROOT, transform=self.train_transform)
        valid_dataset = datasets.ImageFolder(VALID_ROOT, transform=self.valid_transform)

        print('----- train -----')
        print('image qty:', len(train_dataset))
        print('classes  :', train_dataset.classes)

        print('\n----- validation -----')
        print('image qty:', len(valid_dataset))
        print('classes  :', valid_dataset.classes)

        return train_dataset, valid_dataset

    def create_data_loader(self, train_dataset, valid_dataset):
        # データローダーの生成
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader

    def show_img(loader):
        data_iter = iter(loader)
        imgs, labels = next(data_iter)
        img = imgs[0]
        img_permute = img.permute(1, 2, 0)
        img_permute = 0.5 * img_permute + 0.5
        img_permute = np.clip(img_permute, 0, 1)
        plt.imshow(img_permute)

    def train(self, epochs=15):
        train_dataset, valid_dataset = self.load_images(self.image_root)
        train_loader, valid_loader = self.create_data_loader(train_dataset, valid_dataset)

        # モデル生成・GPUに送る
        model = TwiceCNN(len(train_dataset.classes))
        model.to(self.device)

        # 損失関数: 分類なのでクロスエントロピー
        criterion = nn.CrossEntropyLoss()

        # オプティマイザ
        #  - weight_decay: 重み付けが大きくなりすぎないようにL2正則化を行ってくれる
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            running_loss = 0.0
            running_acc = 0.0
            for imgs, labels in train_loader:
                # データをGPUに送る
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # 勾配の初期化
                optimizer.zero_grad()

                # 順伝播
                output = model(imgs)

                # 誤差計算
                loss = criterion(output, labels)
                running_loss += loss.item()

                # 精度計算
                pred = torch.argmax(output, dim=1)
                running_acc += torch.mean(pred.eq(labels).float())

                # 誤差逆伝播
                loss.backward()

                # 重み・バイアス更新
                optimizer.step()
            running_loss /= len(train_loader)
            running_acc /= len(train_loader)
            train_losses.append(running_loss)
            train_accs.append(running_acc)

            # Validation
            running_loss = 0.0
            running_acc = 0.0
            for imgs, labels in valid_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = model(imgs)
                loss = criterion(output, labels)
                running_loss += loss.item()
                pred = torch.argmax(output, dim=1)
                running_acc += torch.mean(pred.eq(labels).float())
            running_loss /= len(valid_loader)
            running_acc /= len(valid_loader)
            val_losses.append(running_loss)
            val_accs.append(running_acc)
            print(f'epoch:{epoch}, loss:{train_losses[epoch]}, acc:{train_accs[epoch]}, val_loss:{running_loss}, val_acc:{running_acc}')

        sns.set()
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='validation')
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

        plt.plot(train_accs, label='train')
        plt.plot(val_accs, label='validation')
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    IMAGE_ROOT = './images'
    classifier = TwiceClassifier(image_root=IMAGE_ROOT)
    classifier.train()
    