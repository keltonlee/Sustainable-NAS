import torch
import torch.nn as nn

from verify_common import print_q15, nchw2nhwc

def main():
    torch.manual_seed(0)
    torch.set_printoptions(threshold=10_000)

    conv = nn.Conv2d(4, 4, 1, groups=4, bias=False)
    inputs = torch.rand(1, 4, 32, 32)
    conv.weight = nn.Parameter(torch.rand(4, 1, 1, 1))

    outputs = conv(inputs)

    print('inputs')
    print_q15(inputs)
    print('inputs NHWC')
    print_q15(nchw2nhwc(inputs))
    print('weight')
    print_q15(conv.weight)
    print('outputs')
    print_q15(outputs)

if __name__ == '__main__':
    main()
