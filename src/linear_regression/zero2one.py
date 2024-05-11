# 线性回归
import random

import torch
import matplotlib
from d2l import torch as d2l


class Zero2One:

    @staticmethod
    def synthetic_data(w, b, num_examples):
        """ 人造数据集 y = Xw + b + 噪声
        synthetic: 合成的，人造的，非天然的，合成物

        :param w: 权重
        :param b: 噪声
        :param num_examples: 样本数
        :return: X, y.reshape((-1, 1)) 特征矩阵，目标向量
        """
        # torch.normal 从给定均值和标准差的正态分布数据中随机生成张量
        # 目的是生成一个张量，张量的值的分配满足 均值为0，标准差为1 的正态分布
        # 张量的形状是 样本数 * 特征数
        X = torch.normal(0, 1, (num_examples, len(w)))

        # torch.matmul 两个张量的矩阵乘积
        y = torch.matmul(X, w) + b

        # 添加噪音
        y += torch.normal(0, 0.01, y.shape)

        # X代表特征矩阵，即包含了模型训练所需的输入特征数据。
        # y.reshape((-1, 1)) 代表目标值向量，即模型训练的目标变量数据。
        # reshape((-1, 1)) 操作，将原始的 目标值向量y 重新调整为一个列向量，确保与特征矩阵X的维度匹配
        return X, y.reshape((-1, 1))

    @staticmethod
    def data_iter(batch_size, features, labels):
        """ 根据参数，从全量特征中随机选取一定尺寸的特征并返回

        :param batch_size: 批量的大小
        :param features:   全量特征
        :param labels:     标签向量
        :return: Generator(features[batch_indices], labels[batch_indices])
        """
        num_examples = len(features)
        indices = list(range(num_examples))
        # 将序列的所有元素随机排序。
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(
                indices[i:min(i + batch_size, num_examples)]
            )
            yield features[batch_indices], labels[batch_indices]

    @staticmethod
    def linreg(X, w, b):
        """ 线性回归模型

        :param X: 特征矩阵
        :param w: 权重向量
        :param b: 偏移量
        :return:  y = <X, w> + b
        """
        return torch.matmul(X, w) + b

    @staticmethod
    def squared_loss(y_hat, y):
        """ 损失函数：均方误差

        :param y_hat:  计算值
        :param y:      实际值
        :return:       二者的均方误差
        """
        # y.reshape(y_hat.shape) 防止二者不对齐
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    @staticmethod
    def sgd(params, lr, batch_size):
        """ 优化算法：小批量随机梯度下降，朝着减少损失的方向更新参数

        :param params:       所有参数 w 和 b
        :param lr:           特征值
        :param batch_size:   批量大小
        """
        with torch.no_grad():
            for param in params:
                # 朝着减少损失的方向更新参数
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    def train(self, w, b, lr, net, loss, num_epochs, batch_size, features, labels):
        """ 训练函数

        :param w:           特征向量
        :param b:           偏移量
        :param lr:          学习率
        :param net:         模型函数
        :param loss:        损失函数
        :param num_epochs:  轮数
        :param batch_size:  批量大小
        :param features:    特征矩阵
        :param labels:      标签集合
        """
        for epoch in range(num_epochs):
            # 从全部数据中随机获取小批量数据
            for X, y in self.data_iter(batch_size, features, labels):
                # net(X, w, b) 使用指定的模型进行预测
                y_hat = net(X, w, b)
                # loss(y_hat, y) 计算预测值的损失
                l = loss(y_hat, y)
                # 计算梯度
                # 因为 l 形状是 (batch_size,1)，而不是一个标量。
                # l 中的所有元素被加到一起，并以此计算关于 [w,b] 的梯度
                l.sum().backward()
                # 使用参数的梯度更新参数
                self.sgd([w, b], lr, batch_size)

            with torch.no_grad():
                train_l = loss(net(features, w, b), labels)
                print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

    def run(self):
        true_w = torch.tensor([2, -3.4])
        true_b = 4.2
        features, labels = self.synthetic_data(true_w, true_b, 1000)

        # 查看样本
        # print(f"feature[0]: {features[0]} | labels[0]: {labels[0]}")

        # 描绘样本分布图
        d2l.set_figsize()
        # plt.scatter: 具有不同标记大小和/或颜色的 y 与 x 的散点图。
        # matplotlib.pyplot.scatter
        # 将张量中的第二列数据提取出来，并将其转换为 NumPy 数组
        # 使用.detach()的主要目的是为了创建一个新的张量，该张量与原始张量不再共享梯度信息，即脱离了计算图。
        d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
        d2l.plt.show()

        batch_size = 10
        # for X, y in self.data_iter(batch_size, features, labels):
        #     print(f"X: {X} | y: {y}")
        #     break

        # 定义初始化模型参数
        w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        lr = 0.03
        num_epochs = 3

        self.train(w, b, lr, self.linreg, self.squared_loss, num_epochs, batch_size, features, labels)

        print(f"w 的估计误差：{true_w - w.reshape(true_w.shape)}")
        print(f"b 的估计误差：{true_b - b}")


if __name__ == '__main__':
    Zero2One().run()
