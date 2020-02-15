
import torch.nn as nn
import torch

# class EncoderNet(nn.Module):
#     configs = [3, 96, 256, 384, 384, 256]

#     def __init__(self, width_mult=1):
#         configs = list(map(lambda x: 3 if x == 3 else
#                        int(x*width_mult), EncoderNet.configs))
#         super(EncoderNet, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(configs[0], configs[1], kernel_size=5, padding=2),
#             nn.BatchNorm2d(configs[1]),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(configs[1], configs[2], kernel_size=5, padding=2),
#             nn.BatchNorm2d(configs[2]),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(configs[2], configs[3], kernel_size=3, padding=1),
#             nn.BatchNorm2d(configs[3]),
#             nn.ReLU(inplace=True),
#             )
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(configs[3], configs[4], kernel_size=3, padding=1),
#             nn.BatchNorm2d(configs[4]),
#             nn.ReLU(inplace=True),
#             )

#         self.layer5 = nn.Sequential(
#             nn.Conv2d(configs[4], configs[5], kernel_size=3, padding=1),
#             nn.BatchNorm2d(configs[5]),
#             )
#         self.feature_size = configs[5]

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x


class EncoderNet(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), EncoderNet.configs))
        super(EncoderNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
            )
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class DecoderNet(nn.Module):
    configs = [256, 256, 96, 64, 4]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), DecoderNet.configs))
        super(DecoderNet, self).__init__()
        self.layer1 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=configs[0],out_channels=configs[1], kernel_size=5),
                    nn.BatchNorm2d(configs[1]),
                    nn.ReLU(inplace=True),
                    )
        self.layer2 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=configs[1],out_channels=configs[2], kernel_size=5, stride=2),
                    nn.BatchNorm2d(configs[2]),
                    nn.ReLU(inplace=True),
                    )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=configs[2],out_channels=configs[3], kernel_size=3, stride=2),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=configs[3],out_channels=configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            )
        self.feature_size = configs[4]
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    test = torch.ones(1, 3, 400, 600)

    model = nn.Sequential(EncoderNet(), DecoderNet())

    test_out = model(test)
    # 185x285
    print(test_out)

