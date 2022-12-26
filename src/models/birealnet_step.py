import math
import os

import numpy as np
from mindspore import Parameter
from mindspore import dtype as mstype
from mindspore import ops, Tensor, nn

from src.models.initializer import uniform_, _calculate_fan_in_and_fan_out
from src.models.initializer import zeros_, kaiming_normal_, constant_

__all__ = ['birealnet18', 'birealnet34']

if os.getenv("DEVICE_TARGET", "GPU") == "GPU" or int(os.getenv("DEVICE_NUM")) == 1:
    BatchNorm2d = nn.BatchNorm2d
elif os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    raise ValueError(f"Model doesn't support devide_num = {int(os.getenv('DEVICE_NUM'))} "
                     f"and device_target = {os.getenv('DEVICE_TARGET')}")


class HardTanh(nn.Cell):
    def __init__(self, max_value=1., min_value=-1):
        super(HardTanh, self).__init__()
        self.max_value = Tensor(max_value, mstype.float32)
        self.min_value = Tensor(min_value, mstype.float32)

    def construct(self, x):
        x = ops.clip_by_value(x, self.min_value, self.max_value)
        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, has_bias=False)


class StrastraightThroughEstimator(nn.Cell):
    '''
    take a real value x
    output sign(x)
    '''

    def __init__(self):
        super(StrastraightThroughEstimator, self).__init__()
        self.sign = ops.Sign()
        self.one = Tensor(1., mstype.float32)

    def construct(self, input):
        """ construct """
        input = input * 1
        return self.sign(input)

    def bprop(self, input, output, grad_output):
        """ bprop """
        # input = input * 1
        # output = output * 1
        dtype = grad_output.dtype
        # mask = ops.LessEqual()(ops.Abs()(input), self.one)
        # grad_input = grad_output * ops.Cast()(self.one, dtype) * ops.Cast()(mask, dtype)
        grad_input = ops.Cast()(grad_output, mstype.float32)
        return (grad_input,)


class BinaryActivation(nn.Cell):
    '''
    take a real value x
    output sign(x)
    '''

    def __init__(self):
        super(BinaryActivation, self).__init__()
        self.sign = ops.Sign()
        self.min_value = Tensor(-1, mstype.float32)
        self.max_value = Tensor(1, mstype.float32)

    def construct(self, input):
        """ construct """
        input = input * 1
        return self.sign(input)

    def bprop(self, input, output, grad_output):
        """ bprop """
        input = input * 1
        output = output * 1
        dtype = grad_output.dtype
        input_clip = ops.clip_by_value(input, self.min_value, self.max_value)
        input_abs = ops.Abs()(input_clip)
        grad_input = grad_output * ops.Cast()(2 - 2 * input_abs, grad_output.dtype)
        return (ops.Cast()(grad_input, dtype),)


class HardBinaryConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(HardBinaryConv, self).__init__()
        self.binarize_op_w = StrastraightThroughEstimator()
        self.binary_activation = BinaryActivation()
        self.bias = None
        self.weight = Parameter(Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size),
                                       mstype.float32) * 0.001, requires_grad=True)
        self.conv2d = ops.Conv2D(out_channel=out_channels, kernel_size=kernel_size, pad_mode="same", stride=stride)

    def construct(self, x):
        if self.training:
            input = self.binary_activation(x / ops.Sqrt()(x.var((1, 2, 3), 0, True)))
        else:
            input = self.binary_activation(x)
        real_weights = self.weight
        binary_weight = self.binarize_op_w(self.weight)
        scaling_factor = ops.ReduceMean(True)(ops.Abs()(real_weights), [1, 2, 3])
        scaling_factor = ops.stop_gradient(scaling_factor)
        output = self.conv2d(input, binary_weight)
        output = output * ops.Cast()(scaling_factor.reshape(1, -1, 1, 1), output.dtype)
        return output


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = BatchNorm2d(planes)
        self.act = HardTanh()

        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.binary_conv(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)
        return out


class BiRealNet(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(BiRealNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = BatchNorm2d(64)
        self.act = HardTanh()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = ops.ReduceMean(keep_dims=True)
        # it can merge into fc
        self.bn2 = BatchNorm2d(512 * block.expansion)
        self.fc = nn.Dense(512 * block.expansion, num_classes)
        self.init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                BatchNorm2d(planes * block.expansion)])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x, (2, 3))
        x = self.bn2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def init_weights(self):
        for name, cell in self.cells_and_names():
            init_weights(cell, name)


def init_weights(cell: nn.Cell, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (cell name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(cell, (nn.Conv2d, nn.Dense, HardBinaryConv)):
        # NOTE conv was left to pytorch default in my original init
        kaiming_normal_(cell.weight)
        if cell.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                uniform_(cell.bias, bound)
    elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        constant_(cell.gamma, value=1)
        zeros_(cell.beta)


def birealnet18(**kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def birealnet34(**kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model


if __name__ == "__main__":
    model = birealnet34(num_classes=1000)
    data = Tensor(np.random.randn(2, 3, 224, 224), dtype=mstype.float32)
    out = model(data)
    print(out.shape)
    params = 0.
    for name, param in model.parameters_and_names():
        params += np.prod(param.shape)
        print(name)
    print(params, 21814696)
