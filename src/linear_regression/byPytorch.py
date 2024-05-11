import torch
import numpy as np
from torch import nn
from torch.utils import data
from d2l import torch as d2l


class ByPorch:

    @staticmethod
    def load_array(data_arrays, batch_size, is_train=True):
        """从给定的 data_arrays 中取出小批量数据

        :param data_arrays:
        :param batch_size:
        :param is_train:
        :return:
        """
        dataset = data.TensorDataset(*data_arrays)
        # shuffle 是否需要打乱顺序
        # 从给定的 dataset 中取出小批量数据
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    def run(self):
        true_w = torch.tensor([2, -3.4])
        true_b = 4.2
        features, labels = d2l.synthetic_data(true_w, true_b, 1000)

        batch_size = 10
        data_iter = self.load_array((features, labels), batch_size)
        # next(iter(data_iter))

        # nn.Sequential: A sequential container. 一个顺序容器。
        # nn.Linear: 对传入数据应用线性变换
        # 初始化模型
        net = nn.Sequential(nn.Linear(2, 1))

        # 初始化模型参数
        net[0].weight.data.normal_(0, 0.01)  # 设置权重
        net[0].bias.data.fill_(0)            # 设置偏移量

        # 均方误差函数
        loss = nn.MSELoss()

        # 实例化 SGD
        trainer = torch.optim.SGD(net.parameters(), lr=0.03)

        num_epochs = 3
        for epoch in range(num_epochs):
            for X, y in data_iter:
                l = loss(net(X), y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
            l = loss(net(features), labels)
            print(f"epoch {epoch + 1}, loss {l:f}")


if __name__ == '__main__':
    ByPorch().run()
