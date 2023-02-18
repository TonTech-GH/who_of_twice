import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from twice_cnn import TwiceCNN


class TwiceClassifier:
    IMG_SIZE = 128

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
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        self.train_transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),  # ランダムに左右反転
            transforms.ColorJitter(),  # ランダムに画像の色調を変更
            transforms.RandomRotation(10),  # ランダムに画像回転(±10度)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def load_images(self, image_root):
        # 画像の読み込み
        TRAIN_ROOT = f'{image_root}/train'
        VALID_ROOT = f'{image_root}/valid'

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

    def train(self, epochs=15, lr=0.0001, weight_decay=5e-4):
        train_dataset, valid_dataset = self.load_images(self.image_root)
        train_loader, valid_loader = self.create_data_loader(train_dataset, valid_dataset)

        # モデル生成・GPUに送る
        model = TwiceCNN(len(train_dataset.classes), self.IMG_SIZE)
        model.to(self.device)

        # 損失関数: 分類なのでクロスエントロピー
        criterion = nn.CrossEntropyLoss()

        # オプティマイザ
        #  - weight_decay: 重み付けが大きくなりすぎないようにL2正則化を行ってくれる
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            running_loss = 0.0
            running_acc = 0.0
            for imgs, labels in tqdm.tqdm(train_loader, desc=f'epoch{epoch+1} train'):
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
            for imgs, labels in tqdm.tqdm(valid_loader, desc=f'epoch{epoch+1} val  '):
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
            print(f'epoch:{epoch+1}, loss:{train_losses[epoch]}, acc:{train_accs[epoch]}, val_loss:{running_loss}, val_acc:{running_acc}')

        # 学習曲線の出力
        plot_data = {
            'loss': {'train': train_losses, 'val': val_losses},
            'accuracy': {'train': train_accs, 'val': val_accs},
        }
        if self.device == 'cuda':
            plot_data['accuracy'] = {'train': [acc.cpu() for acc in train_accs], 'val': [acc.cpu() for acc in val_accs]}

        sns.set()
        plot_num = len(plot_data)
        fig, axes = plt.subplots(plot_num, 1, figsize=(7, 5 * plot_num))

        for ax, (title, data_dict) in zip(axes, plot_data.items()):
            ax.plot(data_dict['train'], label='train')
            ax.plot(data_dict['val'], label='validation')
            ax.set_title(title)
            ax.set_xlabel('epoch')
            ax.set_ylabel(title)
            ax.legend()

        plt.tight_layout()
        fig.savefig(f'result_epochs_{epochs}_batchsize_{self.batch_size}_lr_{lr}_weightdecay_{weight_decay}_imgsize_{self.IMG_SIZE}.png')
        plt.show()


if __name__ == '__main__':
    IMAGE_ROOT = './images_faces'

    # 学習率(0.0001がtrain/validのバランスが良さそう)
    # lr_list = [0.0005, 0.0001, 0.00005]
    lr_list = [0.0001]

    # weight_decay(変えても結果はあまり変わらない)
    # wd_list = [0.0005, 0.0001, 0.001]
    wd_list = [0.0005]

    for lr in lr_list:
        for wd in wd_list:
            print('=' * 30 + f' lr:{lr}, wd:{wd} ' + '=' * 30)
            classifier = TwiceClassifier(image_root=IMAGE_ROOT, batch_size=32)
            classifier.train(epochs=30, lr=lr, weight_decay=wd)
