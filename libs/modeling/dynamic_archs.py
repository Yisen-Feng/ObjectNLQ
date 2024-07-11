import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms
import math
from.meta_archs import PtTransformerRegHead,PtTransformerClsHead,PtTransformer
@register_meta_arch("DynamicPointTransformer")
class DPtTransformer(PtTransformer):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, (
                "Input length must be smaller than max_seq_len during training", max_len, self.max_seq_len)
        # set max_len to self.max_seq_len
        # max_len = self.max_seq_len
        max_len=math.ceil(max_len/512)*512
        # batch input shape B, C, T
        batch_shape = [len(feats), feats[0].shape[0], max_len]
        batched_inputs = feats[0].new_full(batch_shape, padding_val)
        for feat, pad_feat in zip(feats, batched_inputs):
            pad_feat[..., :feat.shape[-1]].copy_(feat)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    