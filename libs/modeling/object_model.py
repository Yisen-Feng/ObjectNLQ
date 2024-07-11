import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms
from .meta_archs import PtTransformer, PtTransformerClsHead, PtTransformerRegHead

from transformers import AutoTokenizer
def iou(reference,gt):
        #reference:[...,2],gt:[...,2]要求两者可广播
        inter_left = torch.max(reference[...,0], gt[...,0])#[batch,T]
        inter_right = torch.min(reference[...,1], gt[...,1])
        inter = torch.relu( inter_right - inter_left)
        union_left = torch.min(reference[...,0], gt[...,0])
        union_right = torch.max(reference[...,1], gt[...,1])
        union = torch.relu( union_right - union_left)
        overlap =  inter / union
        return overlap

@register_meta_arch("ObjectMlp")
class ObjectMlp(PtTransformer):

    def __init__(
            self,
            backbone_type,  # a string defines which backbone we use
            fpn_type,  # a string defines which fpn we use
            backbone_arch,  # a tuple defines # layers in embed / stem / branch
            scale_factor,  # scale factor between branch layers
            input_vid_dim,  # input video feat dim
            input_txt_dim,  # input text feat dim
            max_seq_len,  # max sequence length (used for training)
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            n_head,  # number of heads for self-attention in transformer
            n_mha_win_size,  # window size for self attention; -1 to use full seq
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            fpn_dim,  # feature dim on FPN
            fpn_with_ln,  # if to apply layer norm at the end of fpn
            fpn_start_level,  # start level of fpn
            head_dim,  # feature dim for head
            regression_range,  # regression range on each level of FPN
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            head_with_ln,  # attach layernorm to reg/cls heads
            use_abs_pe,  # if to use abs position encoding
            use_rel_pe,  # if to use rel position encoding
            num_classes,  # number of action classes
            use_lmha_in_fpn,
            object_dim,
            train_cfg,  # other cfg for training
            test_cfg  # other cfg for testing
    ):
        super().__init__(backbone_type,  # a string defines which backbone we use
            fpn_type,  # a string defines which fpn we use
            backbone_arch,  # a tuple defines # layers in embed / stem / branch
            scale_factor,  # scale factor between branch layers
            input_vid_dim,  # input video feat dim
            input_txt_dim,  # input text feat dim
            max_seq_len,  # max sequence length (used for training)
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            n_head,  # number of heads for self-attention in transformer
            n_mha_win_size,  # window size for self attention; -1 to use full seq
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            fpn_dim,  # feature dim on FPN
            fpn_with_ln,  # if to apply layer norm at the end of fpn
            fpn_start_level,  # start level of fpn
            head_dim,  # feature dim for head
            regression_range,  # regression range on each level of FPN
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            head_with_ln,  # attach layernorm to reg/cls heads
            use_abs_pe,  # if to use abs position encoding
            use_rel_pe,  # if to use rel position encoding
            num_classes,  # number of action classes
            use_lmha_in_fpn,
            train_cfg,  # other cfg for training
            test_cfg  # other cfg for testing
            )
        self.object_dim=object_dim
        #更新backbone
        self.backbone = make_backbone(
            # 'convTransformer',
            backbone_type,
            **{
                'n_vid_in': input_vid_dim,
                'n_txt_in': input_txt_dim,
                'n_embd': embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch': backbone_arch,
                'mha_win_size': self.mha_win_size,
                'scale_factor': scale_factor,
                'with_ln': embd_with_ln,
                'attn_pdrop': 0.0,
                'proj_pdrop': self.train_dropout,
                'path_pdrop': self.train_droppath,
                'use_abs_pe': use_abs_pe,
                'use_rel_pe': use_rel_pe,
                'use_lmha_in_fpn':use_lmha_in_fpn,
                'object_dim':object_dim
            }
        )

    def process_object(self,video):
        object_feats = [x['object_feats'] for x in video]
        torch_object_feats=[]
        #确定object_feat的最大长度

        max_seq_len=self.max_seq_len
        #获取mask和object_feat_maxlen
        feats_len=[]
        for object_feat in object_feats:
            if object_feat is None:
                pad_feat_len=torch.zeros(max_seq_len,dtype=torch.int32).to(self.device)
            else:
                feat_len=[]
                for object_f in object_feat:
                    if object_f is None:
                        feat_len.append(0)
                    else:
                        feat_len.append(object_f.shape[0])
                feat_len=torch.tensor(feat_len).to(self.device)
                pad_feat_len=F.pad(feat_len,(0,max_seq_len-len(feat_len)),"constant",0)
            feats_len.append(pad_feat_len)
        feats_len=torch.stack(feats_len,dim=0)#[bs,max_seq_len]
        object_feat_maxlen=max(1,feats_len.max().item())
        batched_masks = (torch.arange(object_feat_maxlen)[None,None, :].to(self.device)) < (feats_len.unsqueeze(-1))#[bs,max_seq_len,maxlen]

        # print(batched_masks)
        #获取padding object_feat
        for object_feat in object_feats:
            if object_feat is None:
                pad_object_feat=torch.zeros((max_seq_len,object_feat_maxlen,self.object_dim)).to(self.device)
            else:
                torch_object_feat=[]
                # if len(object_feat)==0:
                #     print("object_feat为空",[v['query_id'] for v in video])
                for object_f in object_feat:
                    if object_f is None:
                        torch_object_feat.append(torch.zeros(object_feat_maxlen,self.object_dim).to(self.device))
                    else:
                        pad_object_f=F.pad(object_f, (0, 0, 0, object_feat_maxlen-object_f.shape[0]), "constant", 0).to(self.device)
                        torch_object_feat.append(pad_object_f.float())
                # if len(torch_object_feat)==0:
                #     print("torch_object_feat为空",[v['query_id'] for v in video])
                torch_object_feat=torch.stack(torch_object_feat,dim=0)#[T,maxlen,C]torch_object_feat为空[]，说明object_feat为空
                T,_,_=torch_object_feat.shape
                pad_object_feat=F.pad(torch_object_feat,(0,0,0,0,0,max_seq_len-T),"constant",0)
            torch_object_feats.append(pad_object_feat)
        torch_object_feats=torch.stack(torch_object_feats,dim=0).to(self.device)#[bs,max_seq_len,maxlen,C]

        return torch_object_feats,batched_masks
    def forward(self, video_list,**kwargs):
        # video_list:  <class 'list'> 1
        # video_list[0] <class 'dict'>

        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        src_obj,src_obj_mask=self.process_object(video_list)
        src_vid, src_vid_mask = self.preprocessing(video_list)
        src_txt, src_txt_mask = self.query_preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask)
        # print("len(feats): ",len(feats))
        # feats:  <class 'tuple'> 6
        # 以下的1应为bsz
        # 0 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 2560])
        # 1 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 1280])
        # 2 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 640])
        # 3 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 320])
        # 4 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 160])
        # 5 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 80])
        
        #neck仅用于进行可能存在的正则化
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        assert self.num_classes > 0
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]

        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]

            assert video_list[0]['one_hot_labels'] is not None, "GT action labels does not exist"
            gt_labels = [x['one_hot_labels'].to(self.device) for x in video_list]
            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels, self.num_classes)

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets, self.num_classes
            )

            return results
def normal_distribution(x, mu=0, sigma=1):
    return (-(x - mu)**2 / (2 * sigma**2)).exp()    
@register_meta_arch("ObjectLocPointTransformer")
class ObjectPtTransformer(PtTransformer):

    def __init__(
            self,
            backbone_type,  # a string defines which backbone we use
            fpn_type,  # a string defines which fpn we use
            backbone_arch,  # a tuple defines # layers in embed / stem / branch
            scale_factor,  # scale factor between branch layers
            input_vid_dim,  # input video feat dim
            input_txt_dim,  # input text feat dim
            max_seq_len,  # max sequence length (used for training)
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            n_head,  # number of heads for self-attention in transformer
            n_mha_win_size,  # window size for self attention; -1 to use full seq
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            fpn_dim,  # feature dim on FPN
            fpn_with_ln,  # if to apply layer norm at the end of fpn
            fpn_start_level,  # start level of fpn
            head_dim,  # feature dim for head
            regression_range,  # regression range on each level of FPN
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            head_with_ln,  # attach layernorm to reg/cls heads
            use_abs_pe,  # if to use abs position encoding
            use_rel_pe,  # if to use rel position encoding
            num_classes,  # number of action classes
            use_lmha_in_fpn,
            object_dim,
            object_win_size,
            object_use_cross_model,
            train_cfg,  # other cfg for training
            test_cfg  # other cfg for testing
    ):
        super().__init__(backbone_type,  # a string defines which backbone we use
            fpn_type,  # a string defines which fpn we use
            backbone_arch,  # a tuple defines # layers in embed / stem / branch
            scale_factor,  # scale factor between branch layers
            input_vid_dim,  # input video feat dim
            input_txt_dim,  # input text feat dim
            max_seq_len,  # max sequence length (used for training)
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            n_head,  # number of heads for self-attention in transformer
            n_mha_win_size,  # window size for self attention; -1 to use full seq
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            fpn_dim,  # feature dim on FPN
            fpn_with_ln,  # if to apply layer norm at the end of fpn
            fpn_start_level,  # start level of fpn
            head_dim,  # feature dim for head
            regression_range,  # regression range on each level of FPN
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            head_with_ln,  # attach layernorm to reg/cls heads
            use_abs_pe,  # if to use abs position encoding
            use_rel_pe,  # if to use rel position encoding
            num_classes,  # number of action classes
            use_lmha_in_fpn,
            train_cfg,  # other cfg for training
            test_cfg  # other cfg for testing
            )
        self.object_dim=object_dim
        #更新backbone
        self.backbone_type=backbone_type
        self.train_cfg=train_cfg
        
        self.backbone = make_backbone(
            # 'convTransformer',
            backbone_type,
            **{
                'n_vid_in': input_vid_dim,
                'n_txt_in': input_txt_dim,
                'n_embd': embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch': backbone_arch,
                'mha_win_size': self.mha_win_size,
                'scale_factor': scale_factor,
                'with_ln': embd_with_ln,
                'attn_pdrop': 0.0,
                'proj_pdrop': self.train_dropout,
                'path_pdrop': self.train_droppath,
                'use_abs_pe': use_abs_pe,
                'use_rel_pe': use_rel_pe,
                'use_lmha_in_fpn':use_lmha_in_fpn,
                'object_dim':object_dim,
                'object_win_size':object_win_size,
                'object_use_cross_model':object_use_cross_model
            }
        )
        self.multiple_gaussian=False
        if self.multiple_gaussian:
            self.mu = nn.Parameter(torch.zeros(5080, 1), requires_grad=True)
            self.sigma = nn.Parameter(torch.ones(5080, 1), requires_grad=True)
            self.mu_reg_left = nn.Parameter(-torch.ones(5080, 1)*0.5, requires_grad=True)
            self.sigma_reg_left = nn.Parameter(torch.ones(5080, 1), requires_grad=True)
            self.mu_reg_right = nn.Parameter(torch.ones(5080, 1)*0.5, requires_grad=True)
            self.sigma_reg_right = nn.Parameter(torch.ones( 5080,1), requires_grad=True)
        else:
            self.mu = nn.Parameter(torch.zeros(1, 1), requires_grad=True)
            self.sigma = nn.Parameter(torch.ones(1, 1), requires_grad=True)
            self.mu_reg_left = nn.Parameter(-torch.ones(1, 1)*0.5, requires_grad=True)
            self.sigma_reg_left = nn.Parameter(torch.ones(1, 1), requires_grad=True)
            self.mu_reg_right = nn.Parameter(torch.ones(1, 1)*0.5, requires_grad=True)
            self.sigma_reg_right = nn.Parameter(torch.ones( 1,1), requires_grad=True)
        self.norm_cls_loss=True
        self.norm_reg_loss=True
        self.devide_stride=True
        
    def process_object(self,video):
        object_feats = [x['object_feats'] for x in video]
        torch_object_feats=[]
        #确定object_feat的最大长度

        max_seq_len=self.max_seq_len
        #获取mask和object_feat_maxlen
        feats_len=[]
        for object_feat in object_feats:
            if object_feat is None:
                pad_feat_len=torch.zeros(max_seq_len,dtype=torch.int32).to(self.device)
            else:
                feat_len=[]
                for object_f in object_feat:
                    if object_f is None:
                        feat_len.append(0)
                    else:
                        feat_len.append(object_f.shape[0])
                feat_len=torch.tensor(feat_len).to(self.device)
                pad_feat_len=F.pad(feat_len,(0,max_seq_len-len(feat_len)),"constant",0)
            feats_len.append(pad_feat_len)
        feats_len=torch.stack(feats_len,dim=0)#[bs,max_seq_len]
        object_feat_maxlen=max(1,feats_len.max().item())
        batched_masks = (torch.arange(object_feat_maxlen)[None,None, :].to(self.device)) < (feats_len.unsqueeze(-1))#[bs,max_seq_len,maxlen]

        # print(batched_masks)
        #获取padding object_feat
        for object_feat in object_feats:
            if object_feat is None:
                pad_object_feat=torch.zeros((max_seq_len,object_feat_maxlen,self.object_dim)).to(self.device)
            else:
                torch_object_feat=[]
                # if len(object_feat)==0:
                #     print("object_feat为空",[v['query_id'] for v in video])
                for object_f in object_feat:
                    if object_f is None:
                        torch_object_feat.append(torch.zeros(object_feat_maxlen,self.object_dim).to(self.device))
                    else:
                        pad_object_f=F.pad(object_f, (0, 0, 0, object_feat_maxlen-object_f.shape[0]), "constant", 0).to(self.device)
                        torch_object_feat.append(pad_object_f.float())
                # if len(torch_object_feat)==0:
                #     print("torch_object_feat为空",[v['query_id'] for v in video])
                torch_object_feat=torch.stack(torch_object_feat,dim=0)#[T,maxlen,C]torch_object_feat为空[]，说明object_feat为空
                T,_,_=torch_object_feat.shape
                pad_object_feat=F.pad(torch_object_feat,(0,0,0,0,0,max_seq_len-T),"constant",0)
            torch_object_feats.append(pad_object_feat)
        torch_object_feats=torch.stack(torch_object_feats,dim=0).to(self.device)#[bs,max_seq_len,maxlen,C]

        return torch_object_feats,batched_masks
    
    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label, num_classes):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x 2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]#[5080,1]
        reg_targets = torch.stack((left, right), dim=-1)
        dist2center = (right - left) / 2.0
        if self.devide_stride:
            normal_prob_cls = normal_distribution(dist2center / (concat_points[:, 3, None] * lens), self.mu, self.sigma)    # [num_pts, 1]
            normal_prob_reg_left = normal_distribution(dist2center / (concat_points[:, 3, None] * lens), self.mu_reg_left, self.sigma_reg_left)    # [num_pts, 1]
            normal_prob_reg_right = normal_distribution(dist2center / (concat_points[:, 3, None] * lens), self.mu_reg_right, self.sigma_reg_right)    # [num_pts, 1]
        else:
            normal_prob_cls = normal_distribution(dist2center / ( lens), self.mu, self.sigma)    # [num_pts, 1]
            normal_prob_reg_left = normal_distribution(dist2center / ( lens), self.mu_reg_left, self.sigma_reg_left)    # [num_pts, 1]
            normal_prob_reg_right = normal_distribution(dist2center / ( lens), self.mu_reg_right, self.sigma_reg_right)    # [num_pts, 1]
        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius

            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)

            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]

        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # limit the regression range for each location and inside the center radius
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))

        # if there are still more than one ground-truths for one point
        # pick the ground-truth with the shortest duration for the point (easiest to regress)
        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        # make sure that each point can only map with at most one ground-truth
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        # gt_label_one_hot = F.one_hot(gt_label, num_classes).to(reg_targets.dtype)
        gt_label_one_hot = gt_label.to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)

        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets,(normal_prob_cls, normal_prob_reg_left, normal_prob_reg_right) 
    
    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels, num_classes):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)

        gt_cls, gt_offset = [], []
        normal_probs_cls, normal_probs_reg = [], []
        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            assert len(gt_segment) == len(gt_label), (gt_segment, gt_label)
            cls_targets, reg_targets,(normal_prob_cls, normal_prob_reg_left, normal_prob_reg_right)  = self.label_points_single_video(
                concat_points, gt_segment, gt_label, num_classes
            )
            # "cls_targets: " #points, num_classes
            # "reg_targets: " #points, 2
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)
            normal_probs_cls.append(normal_prob_cls)
            normal_probs_reg.append([normal_prob_reg_left, normal_prob_reg_right])

        return gt_cls, gt_offset, normal_probs_cls, normal_probs_reg 
    def losses(
            self, fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets,
            normal_probs_cls, normal_probs_reg 
    ):
        # fpn_masks:F (List) [B, T_i]
        # out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)

        #normal_*:B(list) [sum_T,1]
        normal_probs_cls = torch.cat(normal_probs_cls,dim=1).permute(1,0)        # [b, all_points]
        normal_probs_reg_left = torch.cat([x[0] for x in normal_probs_reg],dim=1).permute(1,0)   # [b, all_points]
        normal_probs_reg_right = torch.cat([x[1] for x in normal_probs_reg],dim=1).permute(1,0)

        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        num_classes = gt_target.shape[-1]

        # optional label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum',
            alpha=self.focal_loss_alpha,
            dim=1
        )
        if self.norm_cls_loss:
            normal_probs_cls[~pos_mask] = 1.0
        else:
            normal_probs_cls=torch.ones_like(normal_probs_cls)
        cls_loss *= normal_probs_cls[valid_mask]
        cls_loss = cls_loss.sum()
        cls_loss /= self.loss_normalizer
        # 2. regression using IoU/GIoU loss (defined on positive samples)
        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
            )
            if not self.norm_reg_loss:
                normal_probs_reg_left = torch.ones_like(normal_probs_reg_left)
                normal_probs_reg_right = torch.ones_like(normal_probs_reg_right)
            reg_loss *= (normal_probs_reg_left[pos_mask] + normal_probs_reg_right[pos_mask]) / 2.0
            reg_loss *= normal_probs_cls[pos_mask]              # for one gaussian
            reg_loss = reg_loss.sum()
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss': cls_loss,
                'reg_loss': reg_loss,
                'final_loss': final_loss}
    def query_preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['query_feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        # batch input shape B, T, C
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
    def forward(self, video_list,**kwargs):
        # video_list:  <class 'list'> 1
        # video_list[0] <class 'dict'>

        # batch the video list into feats (B, C, T) and masks (B, 1, T)

        src_obj,src_obj_mask=self.process_object(video_list)
        src_vid, src_vid_mask = self.preprocessing(video_list)
        src_txt, src_txt_mask = self.query_preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask)
        # print("len(feats): ",len(feats))
        # feats:  <class 'tuple'> 6
        # 以下的1应为bsz
        # 0 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 2560])
        # 1 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 1280])
        # 2 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 640])
        # 3 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 320])
        # 4 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 160])
        # 5 item_feats:  <class 'torch.Tensor'> torch.Size([1, 384, 80])
        
        #neck仅用于进行可能存在的正则化
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        assert self.num_classes > 0
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]

        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]

            assert video_list[0]['one_hot_labels'] is not None, "GT action labels does not exist"
            gt_labels = [x['one_hot_labels'].to(self.device) for x in video_list]
            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets, normal_probs_cls, normal_probs_reg  = self.label_points(
                points, gt_segments, gt_labels, self.num_classes)

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets,
                 normal_probs_cls, normal_probs_reg 
            )
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets, self.num_classes
            )

            return results
    