from time import sleep
import torch.nn as nn


class EPARNNModel(nn.Module):
    def __init__(self, hidden_size=8, seq_size=609):
        super().__init__()

        # LSTM 层
        self.rnn1 = nn.LSTM(
            input_size=5,  # 五种客户的数量
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # LSTM 层
        self.rnn2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # LSTM 层
        self.rnn3 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # 全连接层
        self.linear1 = nn.Linear(hidden_size * seq_size, 4096)
        self.fc1 = nn.ReLU()

        # 全连接层
        self.linear2 = nn.Linear(4096, 1024)
        self.fc2 = nn.ReLU()

        # 投影层
        self.linear3 = nn.Linear(1024, seq_size)

    def forward(self, x):
        # x => RNN
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out, _ = self.rnn3(out)

        # RNN => Flatten
        out = out.reshape(x.size(0), -1)

        # Flatten => Linear1
        out = self.linear1(out)
        out = self.fc1(out)

        # Flatten => Linear1
        out = self.linear2(out)
        out = self.fc2(out)

        # Flatten => Linear2
        out = self.linear3(out)

        return out


class EPADNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 全连接层
        self.linear1 = nn.Linear(3045, 4096)
        self.fc1 = nn.ReLU()

        # 全连接层
        self.linear2 = nn.Linear(4096, 2048)
        self.fc2 = nn.ReLU()

        # 全连接层
        self.linear3 = nn.Linear(2048, 1024)
        self.fc3 = nn.ReLU()

        # 全连接层
        self.linear4 = nn.Linear(1024, 609)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(self.linear1(x))
        x = self.fc2(self.linear2(x))
        x = self.fc3(self.linear3(x))
        x = self.linear4(x)
        return x
