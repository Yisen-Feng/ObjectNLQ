from torch import nn,Tensor
import torch
from .blocks import MaskedConv1D, Scale, LayerNorm
from torch.nn import functional as F
class CAndWRegHead(nn.Module):
    """
    用于预测中心与宽度的偏移量，与原版区别在于可以为负
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        #fpn_feats:n_level*[bs,model,ti],fpn_masks:n_level*[bs,1,ti]
        #return:n_level*[B, 2, T_i]
        assert len(fpn_feats) == len(fpn_masks)
        # assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (self.scale[l](cur_offsets),)

        # fpn_masks remains the same
        return out_offsets
class PerlengthCAndWRegHead(nn.Module):
    """
    用于预测中心与宽度的偏移量，与原版区别在于可以为负
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale=nn.Parameter(
            torch.ones(65, dtype=torch.float32),
            requires_grad=True
        )

        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks,strides):
        #fpn_feats:n_level*[bs,model,ti],fpn_masks:n_level*[bs,1,ti],stride:n_level*[bs,1,ti]
        #return:n_level*[B, 2, T_i]
        assert len(fpn_feats) == len(fpn_masks)
        # assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask,stride) in enumerate(zip(fpn_feats, fpn_masks,strides)):
            cur_out = cur_feat
            stride=stride-1
            stride=torch.min(stride,torch.ones_like(stride)*64)
            stride=torch.max(stride,torch.zeros_like(stride))
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (self.scale[stride.long()]*cur_offsets,)

        # fpn_masks remains the same
        return out_offsets
class PerlayerCAndWRegHead(nn.Module):
    """
    用于预测中心与宽度的偏移量，与原版区别在于可以为负
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            num_decoder_layers=3
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale=nn.Parameter(
            torch.ones((num_decoder_layers,65), dtype=torch.float32),
            requires_grad=True
        )

        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks,strides,layer_idx):
        #fpn_feats:n_level*[bs,model,ti],fpn_masks:n_level*[bs,1,ti],stride:n_level*[bs,1,ti]
        #return:n_level*[B, 2, T_i]
        assert len(fpn_feats) == len(fpn_masks)
        # assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask,stride) in enumerate(zip(fpn_feats, fpn_masks,strides)):
            cur_out = cur_feat
            stride=stride-1
            stride=torch.min(stride,torch.ones_like(stride)*64)
            stride=torch.max(stride,torch.zeros_like(stride))
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (self.scale[layer_idx][stride.long()]*cur_offsets,)

        # fpn_masks remains the same
        return out_offsets
class GFLRegHead(nn.Module):
    """
    用于预测中心与宽度的偏移量，与原版区别在于预测的不是具体的数值而是一个分布
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            distribution_dim=100
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.distribution_dim=distribution_dim
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim,distribution_dim*4, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        #fpn_feats:n_level*[bs,model,ti],fpn_masks:n_level*[bs,1,ti]
        #return:n_level*[B, 2, T_i]
        assert len(fpn_feats) == len(fpn_masks)
        # assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)#[B,200,T_i]
            bs,_,t=cur_offsets.shape
            cur_offsets=cur_offsets.reshape(bs,2,-1,t)
            cur_offsets=torch.softmax(cur_offsets,dim=2)
            weight=torch.arange(-self.distribution_dim,self.distribution_dim).to(cur_offsets.device)
            cur_offsets=cur_offsets*weight[None,None,:,None]
            cur_offsets=cur_offsets.sum(dim=2)#[B,2,T_i]
            out_offsets += (cur_offsets,)

        # fpn_masks remains the same
        return out_offsets
class PerLengthRegHead(nn.Module):
    """
    每一类长度分别回归（[1,2],[3,4][5,8],[9,16],[17,32],[33,64])
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # self.scale = nn.ModuleList()
        # for idx in range(fpn_levels):
        #     self.scale.append(Scale())
        self.scale=nn.Parameter(
            torch.ones(65, dtype=torch.float32),
            requires_grad=True
        )
        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks,points):
        #fpn_feats:n_level*[bs,model,ti],fpn_masks:n_level*[bs,1,ti]
        #return:n_level*[B, 2, T_i]
        assert len(fpn_feats) == len(fpn_masks)
        # assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask,point) in enumerate(zip(fpn_feats, fpn_masks,points)):
            cur_out = cur_feat
            stride=point[...,-1]-1#[bs,ti]
            stride=torch.min(stride,torch.ones_like(stride)*64).unsqueeze(1)#[bs,1,ti]
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)#[bs,2,ti]
            # out_offsets += (F.relu(self.scale[l](cur_offsets)),)
            out_offsets+=(F.relu(self.scale[stride.long()]*cur_offsets),)

        # fpn_masks remains the same
        return out_offsets    