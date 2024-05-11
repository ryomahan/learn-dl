import torch
from IPython import display
from d2l import torch as d2l


class Zero2One:

    @staticmethod
    def softmax(X: "torch.Tensor"):
        """ softmax 算法

        softmax 算法接受一个 K 维的实数向量，并将其转换为一个同样长度的实数向量，
        其中每一个元素的值都在0到1之间，并且所有元素的和为1。这样的输出可以被解释为概率分布。

        :param X: 待分析的张量
        :return:  计算后的结果
        """
        # torch.exp：返回一个新张量，其元素为输入张量 input 的指数。 计算每一项的 e 的 x 次方
        X_exp = torch.exp(X)
        # tensor.sum(求和操作的维度, 求和后是否保持维度不变）
        # 张量会在设置的维度进行求和，求和之后指定的维度会被移除
        # 设置 keepdim=True 之后会保留原有的维度，但是尺度会变为 1
        partition = X_exp.sum(1, keepdim=True)
        # 返回每个 X 的 exp 占总 X 和的比例
        return X_exp / partition

    def net(self, X: "torch.Tensor", W: "torch.Tensor", b: "torch.Tensor"):
        """ softmax 回归模型

        :param X: 特征矩阵
        :param W: 权重矩阵
        :param b: 偏置矩阵
        :return:  结果
        """
        # O = XW + b
        # XW: torch.matmul(X.reshape((-1, W.shape[0])), W)
        # 使用 reshape 将图像展平为向量 批量大小(256) * 输入维数(784)
        return self.softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

    def run(self):
        batch_size = 256
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

        # 输入的图片 28 * 28 通道数为 1
        # softmax 的输入需要是一个向量，所以将图片转换为向量（损失空间信息）
        num_inputs = 784
        num_outputs = 10

        # size 行数：特征数（输入），列数：分类数（输出）
        W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
        b = torch.zeros(num_outputs, requires_grad=True)


if __name__ == '__main__':
    a = torch.tensor([[1,2], [3,4]])
    print(a)
    print(a.exp())

