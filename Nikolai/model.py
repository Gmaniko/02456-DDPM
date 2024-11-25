import torch
from torch import nn



class ScoreNetwork0(nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self, dataset):
        super().__init__()
        chs = [32, 64, 128, 256, 256]
        if dataset == "MNIST":
            p = 1
            self.dims = (1,28,28)
        elif dataset == "CIFAR10":
            p = 0
            self.dims(3,32,32) 
        else:
            raise ValueError("Invalid dataset, please input either 'MNIST' or 'CIFAR10'")
        
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.dims[0]+1, chs[0], kernel_size=3, padding=1),  # (batch, 32, 32, 32)
                nn.LogSigmoid(),  # (batch, 32, 32, 32)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 32, 16, 16)
                nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, 64, 16, 16)
                nn.LogSigmoid(),  # (batch, 64, 16, 16)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 64, 8, 8)
                nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, 128, 8, 8)
                nn.LogSigmoid(),  # (batch, 128, 8, 8)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=p),  # (batch, 128, 4, 4)
                nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, 256, 4, 4)
                nn.LogSigmoid(),  # (batch, 256, 4, 4)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 256, 2, 2)
                nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, 256, 2, 2)
                nn.LogSigmoid(),  # (batch, 256, 2, 2)
            ),
        ])
        self._tconvs = nn.ModuleList([
            nn.Sequential(
                # input is the output of convs[4]
                nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 256, 4, 4)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding= 1-p),  # (batch, 128, 8, 8)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 16, 16)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 32, 32, 32)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, 32, 32, 32)
                nn.LogSigmoid(),
                nn.Conv2d(chs[0], self.dims[0], kernel_size=3, padding=1),  # (batch, 3, 28, 28)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat((x, t), dim=-3)
        signal = xt
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)

        return signal