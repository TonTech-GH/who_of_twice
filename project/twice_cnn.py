import torch.nn as nn


class TwiceCNN(nn.Module):
    def __init__(self, num_classes, image_size):
        super().__init__()

        # 畳み込み層
        self.features = nn.Sequential(
            # 畳み込み
            #  - in_channels: RGBの3チャネル
            #  - out_channels: 任意のチャネル数
            #  - kernel_size: フィルターの縦横ピクセル数
            #  - padding: パディングで埋めるピクセル数(フィルターが5pxで2px収縮するのでその分2px埋めている)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # プーリング
            #  - kernel_size: 縦横このピクセル数の範囲の最大値をとる(2なら画像の縦横サイズが半分になる)
            nn.MaxPool2d(kernel_size=2),  # image_size / 2

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # image_size / 4

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # image_size / 8

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 全結合層
        input_size = image_size // 8
        self.classifier = nn.Linear(in_features=input_size * input_size * 128, out_features=num_classes)

    def forward(self, x):
        # 畳み込み層
        #  - 出力のsizeは(batch_size, C, H, W) = (32, 128, 4, 4)
        x = self.features(x)
        # ベクトル化
        #  - バッチサイズを指定して残りの次元を1次元にまとめている
        x = x.view(x.size(0), -1)
        # 全結合層
        return self.classifier(x)


if __name__ == '__main__':
    cnn = TwiceCNN(9)
