import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class FPNEfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None, model_name='efficientnet-b3', fpn_channel=128):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.model_name = model_name
        self.fpn_c = fpn_channel
        self._conv_idx = {
            'efficientnet-b0': [15, 10, 4, 2],
            'efficientnet-b3': [25, 17, 7, 4],
            'efficientnet-b5': [38, 26, 12, 7]
        }

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))


        # Lateral layers
        fpn_layers = self.make_fpn(self.model_name)
        self.latlayer1 = fpn_layers[0]
        self.latlayer2 = fpn_layers[1]
        self.latlayer3 = fpn_layers[2]
        self.latlayer4 = fpn_layers[3]

        # Top-down layers
        self.toplayer1 = fpn_layers[4]
        self.toplayer2 = fpn_layers[5]
        self.toplayer3 = fpn_layers[6]

        # Boundary-regressor layers
        self.max_pool = fpn_layers[7]
        self.regression_head = fpn_layers[8]

        self.calibration = {'cdx':0.0, 'cdy':0.0, 'cdw': 0.0, 'cdh': 0.0}


    def make_fpn(self, model_name):
        # Lateral layers

        if model_name == 'efficientnet-b0': # 56 x 56
            latlayer1 = nn.Conv2d(320, self.fpn_c, kernel_size=1, stride=1, padding=0)
            latlayer2 = nn.Conv2d(112, self.fpn_c, kernel_size=1, stride=1, padding=0)
            latlayer3 = nn.Conv2d(40, self.fpn_c//2, kernel_size=1, stride=1, padding=0)
            latlayer4 = nn.Conv2d(24, self.fpn_c//2, kernel_size=1, stride=1, padding=0)

        elif model_name == 'efficientnet-b3': # 75 x 75
            latlayer1 = nn.Conv2d(384, self.fpn_c, kernel_size=1, stride=1, padding=0)
            latlayer2 = nn.Conv2d(136, self.fpn_c, kernel_size=1, stride=1, padding=0)
            latlayer3 = nn.Conv2d(48, self.fpn_c//2, kernel_size=1, stride=1, padding=0)
            latlayer4 = nn.Conv2d(32, self.fpn_c//2, kernel_size=1, stride=1, padding=0)

        elif model_name == 'efficientnet-b5': # 114 x 114
            latlayer1 = nn.Conv2d(512, self.fpn_c, kernel_size=1, stride=1, padding=0)
            latlayer2 = nn.Conv2d(176, self.fpn_c, kernel_size=1, stride=1, padding=0)
            latlayer3 = nn.Conv2d(64, self.fpn_c//2, kernel_size=1, stride=1, padding=0)
            latlayer4 = nn.Conv2d(40, self.fpn_c//2, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        toplayer1 = nn.Conv2d(self.fpn_c, self.fpn_c//2, kernel_size=3, stride=1, padding=1)
        toplayer2 = nn.Conv2d(self.fpn_c//2, self.fpn_c//2, kernel_size=3, stride=1, padding=1)
        toplayer3 = nn.Conv2d(self.fpn_c//2, self.fpn_c//2, kernel_size=3, stride=1, padding=1)

        # Box regressor layers
        if model_name == 'efficientnet-b0':
            max_pool = nn.MaxPool1d(56, stride=1)
            regression_head = self._make_head(56)
        if model_name == 'efficientnet-b3':
            max_pool = nn.MaxPool1d(75, stride=1)
            regression_head = self._make_head(75)
        if model_name == 'efficientnet-b5':
            max_pool = nn.MaxPool1d(114, stride=1)
            regression_head = self._make_head(114)

        return latlayer1, latlayer2, latlayer3, latlayer4, toplayer1, toplayer2, toplayer3, max_pool, regression_head

    def _make_head(self, resolution):

        layers = []
        layers.append(nn.Conv1d(self.fpn_c//2, self.fpn_c//2, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv1d(self.fpn_c//2, self.fpn_c//4, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv1d(self.fpn_c//4, self.fpn_c//4, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))
        layers.append(Flatten())
        layers.append(nn.Linear(self.fpn_c//4 * resolution, 2))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def boundary_regression(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = x_h.view(-1, x_h.shape[2], x_h.shape[3])
        x_h = self.max_pool(x_h)
        x_h = x_h.view(x.shape[0], x.shape[2], -1)
        x_h = x_h.permute(0, 2, 1).contiguous()
        out_h = self.regression_head(x_h)

        x_v = x.permute(0, 3, 1, 2).contiguous()
        x_v = x_v.view(-1, x_v.shape[2], x_v.shape[3])
        x_v = self.max_pool(x_v)
        x_v = x_v.view(x.shape[0], x.shape[3], -1)
        x_v = x_v.permute(0, 2, 1).contiguous()
        out_v = self.regression_head(x_v)

        out = torch.stack([out_h[:, 0], out_v[:, 0], out_h[:, 1], out_v[:, 1]], dim=1)
        return out

    def extract_features(self, inputs, model_name):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        conv_idx = self._conv_idx[model_name]
        convs = []
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx in conv_idx:
                convs.append(x)

        return convs


    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        c2, c3, c4, c5 = self.extract_features(inputs, self.model_name)

        # Top-down
        p5 = self.latlayer1(c5)  # 256x7x7
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)  # 256x14x14
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)  # 256x28x28
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)

        # Box regression
        out = self.boundary_regression(p2)
        return out

    def count_parameters(self):
        for idx, p in enumerate(self.parameters()):
            print(idx, p.numel())
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000000.0

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return FPNEfficientNet(blocks_args, global_params, model_name)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = FPNEfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = FPNEfficientNet(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))