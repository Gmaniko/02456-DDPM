import torch
from torch import nn


class ScoreNetwork0(nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 256]
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                nn.LogSigmoid(),  # (batch, 8, 28, 28)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, ch, 4, 4)
                nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = nn.ModuleList([
            nn.Sequential(
                # input is the output of convs[4]
                nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                nn.LogSigmoid(),
                nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))  # (..., ch0, 28, 28)
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)  # (..., 1, 28, 28)
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t
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
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal
