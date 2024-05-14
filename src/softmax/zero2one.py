import torch
from IPython import display
from d2l import torch as d2l


class Animator:

    def __init__(
        self, xlabel=None, ylabel=None, legend=None,
        xlim=None, ylim=None, xscale="linear", yscale="linear",
        fmts=("-", "m--", "g-", "r:"), nrows=1, ncols=1,
        figsize=(3.5, 2.5)
    ):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # display.display(self.fig)
        # display.clear_output(wait=True)
        self.fig.show()


class Accumulator:
    """ 分类累加器

    可以对不同 index 传入的 args 进行累加并存储在 self.data[index] 中
    """

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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

    @staticmethod
    def cross_entropy(y_hat, y):
        """ 交叉熵损失函数

        :param y_hat:
        :param y:
        :return:
        """
        # y 是一个一维张量，每一项代表在第 index 个预测结果中第 y[index] 个分类是正确结果
        # y_hat 是某次预测的结果张量
        # y_hat[range(len(y_hat)), y] 可以取出每个样本正确分类对应的预测概率的值
        # torch.log 返回具有 input 元素的自然对数的新张量。
        return -torch.log(y_hat[range(len(y_hat)), y])

    @staticmethod
    def accuracy(y_hat: "torch.Tensor", y: "torch.Tensor"):
        """ 找出预测正确的数量 """
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            # 将每一行中值最大的下表存入 y_hat
            y_hat = y_hat.argmax(axis=1)
        # y_hat.type(y.dtype) 将 y_hat 类型与 y 对齐
        cmp = y_hat.type(y.dtype) == y
        # cmp.type(y.dtype) 将 cmp 类型与 y 对齐
        return float(cmp.type(y.dtype).sum())

    @staticmethod
    def updater(W, b, lr, batch_size):
        return d2l.sgd([W, b], lr, batch_size)

    def evaluate_accuracy(self, W, b, net, data_iter):
        """ 计算在指定数据集上模型的精度

        :param net:         模型
        :param data_iter:   数据集的数据生成器
        :return:            模型准确率
        """
        if isinstance(net, torch.nn.Module):
            net.eval()  # 将模型设置为评估模式
        # 累加器
        metric = Accumulator(2)
        for X, y in data_iter:
            # self.accuracy(net(X), y) 计算预测正确的样本数
            metric.add(self.accuracy(net(X, W, b), y), y.numel())
        # metric[0] 分类正确的样本数
        # metric[1] 总样本数
        return metric[0] / metric[1]

    def train_epoch_ch3(self, W, b, lr, batch_size, net, train_iter, loss, updater):
        """ 训练一次的方法 """
        if isinstance(net, torch.nn.Module):
            net.train()
        metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X, W, b)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.backward()
                updater.step()
                metric.add(
                    float(l) * len(y),
                    self.accuracy(y_hat, y),
                    y.size().numel()
                )
            else:
                l.sum().backward()
                updater(W, b, lr, batch_size)
                metric.add(float(l.sum()), self.accuracy(y_hat, y), y.numel())

        # metric[0] / metric[2] loss / 样本数
        # metric[1] / metric[2] 分类正确的样本数 / 样本数
        return metric[0] / metric[2], metric[1] / metric[2]

    def train_ch3(self, W, b, lr, batch_size, net, train_iter, test_iter, loss, num_epochs, updater):
        """ 训练函数 """
        animator = Animator(xlabel="epoch", xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=["train loss", "train acc", "test acc"])

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch_ch3(W, b, lr, batch_size, net, train_iter, loss, updater)
            test_acc = self.evaluate_accuracy(W, b, net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        # train_loss, train_acc = train_metrics

    def predict_ch3(self, W, b, net, test_iter, n=6):
        for X, y in test_iter:
            break

        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(net(X, W, b).argmax(axis=1))
        titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
        d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

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

        lr = 0.1
        num_epochs = 10
        # self.train_ch3(W, b, lr, batch_size, self.net, train_iter, test_iter, self.cross_entropy, num_epochs, self.updater)

        self.predict_ch3(W, b, self.net, test_iter)


if __name__ == '__main__':
    Zero2One().run()

