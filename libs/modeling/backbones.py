import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D, LayerNorm,LMD,zeroLinearLayer,zeroMaskedConv1D,ObjectTransformerBlock,ObjectQueryTransformerBlock,ObjectPlTransformerBlock,ObjectCAonlyTransformerBlock,ObjectShareTransformerBlock,ObjectBMDTransformerBlock)


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,
            n_vid_in,  # input video feature dimension
            n_txt_in,  # input text feature dimension
            n_embd,  # embedding dimension (after convolution)
            n_head,  # number of head for self-attention in transformers
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch=(2, 2, 2, 0, 5),  # (#convs, #stem transformers, #branch transformers)
            mha_win_size=[-1] * 6,  # size of local window for mha
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=False,  # if to attach layernorm after conv
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # droput rate for drop path
            use_abs_pe=False,  # use absolute position embedding
            use_rel_pe=False,  # use relative position embedding
            use_lmha_in_fpn=True,  # use local mha in fpn
            position_type='none'#即不生成提供给dino的位置，single则为生成单个位置
    ):
        super().__init__()
        assert len(arch) == 5
        assert len(mha_win_size) == (1 + arch[3] + arch[4])
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # vid_embedding network using convs
        self.vid_embd = nn.ModuleList()
        self.vid_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_vid_in
            else:
                in_channels = n_embd
            self.vid_embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.vid_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.vid_embd_norm.append(nn.Identity())

        # txt_embedding network using linear projection
        self.txt_embd = nn.ModuleList()
        self.txt_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_txt_in
            else:
                in_channels = n_embd
            self.txt_embd.append(MaskedConv1D(
                in_channels, n_embd, 1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.txt_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.txt_embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )

        self.txt_stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.txt_stem.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=-1,
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=False,
            )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        if use_lmha_in_fpn:
            for idx in range(arch[3]):
                self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=True,
                )
            )

            for idx in range(arch[4]):
                self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=False,
                )
            )
        else:
            for idx in range(arch[3]):
                self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=1,
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=True,
                )
            )

            for idx in range(arch[4]):
                self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=1,
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=False,
                )
            )
        self.position_type=position_type
        if position_type=='single':
            self.LMD=LMD(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=1,
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=True,
                )
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()

        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)

        # vid_stem transformer
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)

        # main branch with downsampling
        if self.position_type=='single':
            query_num=src_query_mask.sum(dim=-1).unsqueeze(-1)
            query=(src_query.sum(dim=-1).unsqueeze(-1))/query_num
            query_mask=query_num>0
            query,_,attn=self.LMD(src_vid,src_vid_mask,query,query_mask)#[bs,C,1]
            query=query.squeeze(-1)#[bs,C]
            return out_feats, out_masks,query,attn
        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks

@register_backbone("prvgTransformer")
class PrvgTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,
            n_vid_in,  # input video feature dimension
            n_txt_in,  # input text feature dimension
            n_embd,  # embedding dimension (after convolution)
            n_head,  # number of head for self-attention in transformers
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch=(2, 2, 2, 0, 5),  # (#convs, #stem transformers, #branch transformers)
            mha_win_size=[-1] * 6,  # size of local window for mha
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=False,  # if to attach layernorm after conv
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # droput rate for drop path
            use_abs_pe=False,  # use absolute position embedding
            use_rel_pe=False,  # use relative position embedding
            use_lmha_in_fpn=True,  # use local mha in fpn
    ):
        super().__init__()
        assert len(arch) == 5
        assert len(mha_win_size) == (1 + arch[3] + arch[4])
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # vid_embedding network using convs
        self.vid_embd = nn.ModuleList()
        self.vid_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_vid_in
            else:
                in_channels = n_embd
            self.vid_embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.vid_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.vid_embd_norm.append(nn.Identity())

        # txt_embedding network using linear projection
        self.txt_embd = nn.ModuleList()
        self.txt_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_txt_in
            else:
                in_channels = n_embd
            self.txt_embd.append(MaskedConv1D(
                in_channels, n_embd, 1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.txt_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.txt_embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )

        self.txt_stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.txt_stem.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=-1,
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=False,
            )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        if use_lmha_in_fpn:
            for idx in range(arch[3]):
                self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=True,
                )
            )

            for idx in range(arch[4]):
                self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=False,
                )
            )
        else:
            for idx in range(arch[3]):
                self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=1,
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=True,
                )
            )

            for idx in range(arch[4]):
                self.branch.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=1,
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=False,
                )
            )
        self.LMD=LMD(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=1,
                    use_rel_pe=self.use_rel_pe,
                    use_cross_modal=True,
                )
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()

        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)

        # vid_stem transformer
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)

        # main branch with downsampling
        query,_=self.LMD(src_vid,src_vid_mask,src_query,src_query_mask)#[bs,C,txt_len]
        
        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)
        
        
        return out_feats, out_masks,query
@register_backbone("ObjectMlpTransformer")
class ObjectMlpTransformerBackbone(ConvTransformerBackbone):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,object_dim=512,
            **kwargs
    ):
        super(ObjectMlpTransformerBackbone,self).__init__(**kwargs)
        n_embd=kwargs['n_embd']
        self.object_mlp=zeroLinearLayer(object_dim,n_embd,layer_norm=False,dropout=0,relu=False)
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()

        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            if idx==0:
                mask_src_obj=src_obj*(src_obj_mask.unsqueeze(-1))#[bs,max_seq_len,maxlen,C]
                src_obj_num=torch.max(src_obj_mask.sum(-1),torch.tensor(1))#[bs,max_seq_len]
                mean_src_obj=mask_src_obj.sum(-2)/(src_obj_num.unsqueeze(-1))#[bs,max_seq_len,C]
                src_obj=self.object_mlp(mean_src_obj)#[bs,T,C]
                
                src_obj=src_obj.permute(0,2,1)
                src_obj=src_obj*src_vid_mask
                src_vid=src_vid+src_obj
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)

        # vid_stem transformer
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)


        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks
from einops import rearrange
@register_backbone("ObjectAttentionTransformer")
class ObjectAttentionTransformerBackbone(ConvTransformerBackbone):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,object_dim=512,
            **kwargs
    ):
        super(ObjectAttentionTransformerBackbone,self).__init__(**kwargs)
        n_embd=kwargs['n_embd']
        arch=kwargs['arch']
        n_head=kwargs['n_head']
        attn_pdrop=kwargs['attn_pdrop']
        proj_pdrop=kwargs['proj_pdrop']
        path_pdrop=kwargs['path_pdrop']
        n_embd_ks=kwargs['n_embd_ks']
        with_ln=kwargs['with_ln']
        self.obj_embd = nn.ModuleList()
        self.obj_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = object_dim
            else:
                in_channels = n_embd
            self.obj_embd.append(MaskedConv1D(
                in_channels=in_channels, out_channels=n_embd, kernel_size=1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.obj_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.obj_embd_norm.append(nn.Identity())

        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(ObjectPlTransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()
        
        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))
        #obj_embedding
        src_obj=rearrange(src_obj,"b t o c -> (b o) c t")
        src_obj_mask=rearrange(src_obj_mask,"b t o -> (b o) 1 t")
        for idx in range(len(self.obj_embd)):
            src_obj, src_obj_mask = self.obj_embd[idx](src_obj, src_obj_mask)
            src_obj = self.relu(self.obj_embd_norm[idx](src_obj))
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)

        # vid_stem transformer
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_obj,src_obj_mask,src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)


        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks

@register_backbone("ObjectTransformer")
class ObjectTransformerBackbone(ConvTransformerBackbone):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,object_dim=512,object_win_size=1,object_use_cross_model=False,
            **kwargs
    ):
        super(ObjectTransformerBackbone,self).__init__(**kwargs)
        n_embd=kwargs['n_embd']
        arch=kwargs['arch']
        n_head=kwargs['n_head']
        attn_pdrop=kwargs['attn_pdrop']
        proj_pdrop=kwargs['proj_pdrop']
        path_pdrop=kwargs['path_pdrop']
        n_embd_ks=kwargs['n_embd_ks']
        with_ln=kwargs['with_ln']
        self.obj_embd = nn.ModuleList()
        self.obj_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = object_dim
                # out_channels=4*n_embd
            # elif idx==arch[0]-1:
            #     in_channels = 4*n_embd
                # out_channels=n_embd
            else:
                in_channels = n_embd
                # out_channels=4*n_embd
            self.obj_embd.append(MaskedConv1D(
                in_channels=in_channels, out_channels=n_embd, kernel_size=1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.obj_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.obj_embd_norm.append(nn.Identity())
        self.obj_stem=nn.ModuleList()
        for idx in range(arch[2]):
            self.obj_stem.append(ObjectQueryTransformerBlock(
                n_embd, n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=object_win_size,
                use_cross_modal=object_use_cross_model,
            )
            )

        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(ObjectPlTransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()
        
        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))
        #obj_embedding
        src_obj=rearrange(src_obj,"b t o c -> (b o) c t")
        src_obj_mask=rearrange(src_obj_mask,"b t o -> (b o) 1 t")
        for idx in range(len(self.obj_embd)):
            src_obj, src_obj_mask = self.obj_embd[idx](src_obj, src_obj_mask)
            src_obj = self.relu(self.obj_embd_norm[idx](src_obj))
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)
        #obj_stem
        hidden_dim=src_query.shape[1]
        src_obj=src_obj.view(B,-1,hidden_dim,T)
        for idx in range(len(self.obj_stem)):
            src_obj, src_obj_mask = self.obj_stem[idx](src_obj,src_obj_mask,src_query, src_query_mask)
        src_obj=src_obj.view(-1,hidden_dim,T)
        # vid_stem transformer
        
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_obj,src_obj_mask,src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)


        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks
    
@register_backbone("ObjectAddTransformer")
class ObjectAddTransformerBackbone(ConvTransformerBackbone):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,object_dim=512,object_win_size=1,object_use_cross_model=False,
            **kwargs
    ):
        super(ObjectAddTransformerBackbone,self).__init__(**kwargs)
        n_embd=kwargs['n_embd']
        arch=kwargs['arch']
        n_head=kwargs['n_head']
        attn_pdrop=kwargs['attn_pdrop']
        proj_pdrop=kwargs['proj_pdrop']
        path_pdrop=kwargs['path_pdrop']
        n_embd_ks=kwargs['n_embd_ks']
        with_ln=kwargs['with_ln']
        self.obj_embd = nn.ModuleList()
        self.obj_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = object_dim
                out_channels=4*n_embd
            elif idx==arch[0]-1:
                in_channels = 4*n_embd
                out_channels=n_embd
            else:
                in_channels = 4*n_embd
                out_channels=4*n_embd
            self.obj_embd.append(MaskedConv1D(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.obj_embd_norm.append(
                    LayerNorm(out_channels)
                )
            else:
                self.obj_embd_norm.append(nn.Identity())
        # self.obj_stem=nn.ModuleList()
        # for idx in range(arch[2]):
        #     self.obj_stem.append(ObjectQueryTransformerBlock(
        #         n_embd, n_head,
        #         attn_pdrop=attn_pdrop,
        #         proj_pdrop=proj_pdrop,
        #         path_pdrop=path_pdrop,
        #         mha_win_size=object_win_size,
        #         use_cross_modal=object_use_cross_model,
        #     )
        #     )

        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(ObjectPlTransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()
        
        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))
        #obj_embedding
        src_obj=rearrange(src_obj,"b t o c -> (b o) c t")
        src_obj_mask=rearrange(src_obj_mask,"b t o -> (b o) 1 t")
        for idx in range(len(self.obj_embd)):
            src_obj, src_obj_mask = self.obj_embd[idx](src_obj, src_obj_mask)
            src_obj = self.relu(self.obj_embd_norm[idx](src_obj))
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)
        #obj_stem
        # hidden_dim=src_query.shape[1]
        # src_obj=src_obj.view(B,-1,hidden_dim,T)
        # for idx in range(len(self.obj_stem)):
        #     src_obj, src_obj_mask = self.obj_stem[idx](src_obj,src_obj_mask,src_query, src_query_mask)
        # src_obj=src_obj.view(-1,hidden_dim,T)
        # vid_stem transformer
        
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_obj,src_obj_mask,src_query, src_query_mask)
        # src_obj=src_obj.view(B,-1,hidden_dim,T)
        # O=src_obj.shape[1]
        # src_obj_mask=src_obj_mask.view(B,O,1,T)
        # src_obj_sum=src_obj.sum(dim=1)
        # src_obj_num=torch.max(torch.tensor(1),src_obj_mask.sum(dim=1))
        # src_obj_mean=src_obj_sum/src_obj_num
        # src_vid=src_vid+src_obj_mean
        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)


        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks
    

@register_backbone("ObjectAllTransformer")
class ObjectAllTransformerBackbone(ConvTransformerBackbone):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,object_dim=512,object_win_size=1,object_use_cross_model=False,
            **kwargs
    ):
        super(ObjectAllTransformerBackbone,self).__init__(**kwargs)
        n_embd=kwargs['n_embd']
        arch=kwargs['arch']
        n_head=kwargs['n_head']
        attn_pdrop=kwargs['attn_pdrop']
        proj_pdrop=kwargs['proj_pdrop']
        path_pdrop=kwargs['path_pdrop']
        n_embd_ks=kwargs['n_embd_ks']
        with_ln=kwargs['with_ln']
        self.obj_embd = nn.ModuleList()
        self.obj_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = object_dim
                out_channels=n_embd
            else:
                in_channels = n_embd
                out_channels=n_embd
            self.obj_embd.append(MaskedConv1D(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.obj_embd_norm.append(
                    LayerNorm(out_channels)
                )
            else:
                self.obj_embd_norm.append(nn.Identity())
        self.obj_stem=nn.ModuleList()
        for idx in range(arch[2]):
            self.obj_stem.append(ObjectCAonlyTransformerBlock(
                n_embd, n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=object_win_size,
                use_cross_modal=object_use_cross_model,
            )
            )

        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(ObjectTransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()
        
        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))
        #obj_embedding
        src_obj=rearrange(src_obj,"b t o c -> (b o) c t")
        src_obj_mask=rearrange(src_obj_mask,"b t o -> (b o) 1 t")
        for idx in range(len(self.obj_embd)):
            src_obj, src_obj_mask = self.obj_embd[idx](src_obj, src_obj_mask)
            src_obj = self.relu(self.obj_embd_norm[idx](src_obj))
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)
        #obj_stem
        hidden_dim=src_query.shape[1]
        src_obj=src_obj.view(B,-1,hidden_dim,T)
        for idx in range(len(self.obj_stem)):
            src_obj, src_obj_mask = self.obj_stem[idx](src_obj,src_obj_mask,src_query, src_query_mask)
        src_obj=src_obj.view(-1,hidden_dim,T)
        # vid_stem transformer
        
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_obj,src_obj_mask,src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)


        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks
    
@register_backbone("ObjectAllShareTransformer")
class ObjectAllShareTransformerBackbone(ConvTransformerBackbone):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,object_dim=512,object_win_size=1,object_use_cross_model=False,
            **kwargs
    ):
        super(ObjectAllShareTransformerBackbone,self).__init__(**kwargs)
        n_embd=kwargs['n_embd']
        arch=kwargs['arch']
        n_head=kwargs['n_head']
        attn_pdrop=kwargs['attn_pdrop']
        proj_pdrop=kwargs['proj_pdrop']
        path_pdrop=kwargs['path_pdrop']
        n_embd_ks=kwargs['n_embd_ks']
        with_ln=kwargs['with_ln']
        self.obj_embd = nn.ModuleList()
        self.obj_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = object_dim
                out_channels=4*n_embd
            elif idx==arch[0]-1:
                in_channels = 4*n_embd
                out_channels=n_embd
            else:
                in_channels = 4*n_embd
                out_channels=4*n_embd
            self.obj_embd.append(MaskedConv1D(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.obj_embd_norm.append(
                    LayerNorm(out_channels)
                )
            else:
                self.obj_embd_norm.append(nn.Identity())
        self.obj_stem=nn.ModuleList()
        for idx in range(2):#无残差就两层好了
            self.obj_stem.append(ObjectCAonlyTransformerBlock(
                n_embd, n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=object_win_size,
                use_cross_modal=object_use_cross_model,
            )
            )

        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(ObjectShareTransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()
        
        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))
        #obj_embedding
        src_obj=rearrange(src_obj,"b t o c -> (b o) c t")
        src_obj_mask=rearrange(src_obj_mask,"b t o -> (b o) 1 t")
        for idx in range(len(self.obj_embd)):
            # src_obj, src_obj_mask = self.obj_embd[idx](src_obj, src_obj_mask)#embeding就开始share
            # src_obj = self.relu(self.obj_embd_norm[idx](src_obj))
            src_obj, src_obj_mask = self.txt_embd[idx](src_obj, src_obj_mask)#embeding就开始share
            src_obj = self.relu(self.txt_embd_norm[idx](src_obj))
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)
        #obj_stem
        hidden_dim=src_query.shape[1]
        src_obj=src_obj.view(B,-1,hidden_dim,T)
        for idx in range(len(self.obj_stem)):#已修改为无残差连接
            src_obj, src_obj_mask = self.obj_stem[idx](src_obj,src_obj_mask,src_query, src_query_mask)
        src_obj=src_obj.view(-1,hidden_dim,T)
        # vid_stem transformer
        
        for idx in range(len(self.vid_stem)):#用参数共享策略
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_obj,src_obj_mask,src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)


        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks
    
@register_backbone("ObjectCATransformer")
class ObjectCATransformerBackbone(ConvTransformerBackbone):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,object_dim=512,object_win_size=1,object_use_cross_model=False,
            **kwargs
    ):
        super(ObjectCATransformerBackbone,self).__init__(**kwargs)
        n_embd=kwargs['n_embd']
        arch=kwargs['arch']
        n_head=kwargs['n_head']
        attn_pdrop=kwargs['attn_pdrop']
        proj_pdrop=kwargs['proj_pdrop']
        path_pdrop=kwargs['path_pdrop']
        n_embd_ks=kwargs['n_embd_ks']
        with_ln=kwargs['with_ln']
        self.obj_embd = nn.ModuleList()
        self.obj_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = object_dim
                # out_channels=4*n_embd
            # elif idx==arch[0]-1:
            #     in_channels = 4*n_embd
                # out_channels=n_embd
            else:
                in_channels = n_embd
                # out_channels=4*n_embd
            self.obj_embd.append(MaskedConv1D(
                in_channels=in_channels, out_channels=n_embd, kernel_size=1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.obj_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.obj_embd_norm.append(nn.Identity())
        self.obj_stem=nn.ModuleList()
        for idx in range(arch[2]):
            self.obj_stem.append(ObjectCAonlyTransformerBlock(
                n_embd, n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=object_win_size,
                use_cross_modal=object_use_cross_model,
            )
            )

        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(ObjectPlTransformerBlock(
            # self.vid_stem.append(ObjectTransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()
        
        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))
        #obj_embedding
        src_obj=rearrange(src_obj,"b t o c -> (b o) c t")
        src_obj_mask=rearrange(src_obj_mask,"b t o -> (b o) 1 t")
        for idx in range(len(self.obj_embd)):
            src_obj, src_obj_mask = self.obj_embd[idx](src_obj, src_obj_mask)
            src_obj = self.relu(self.obj_embd_norm[idx](src_obj))
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        # txt_stem transformer
        for idx in range(len(self.txt_stem)):
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)
        #obj_stem
        hidden_dim=src_query.shape[1]
        src_obj=src_obj.view(B,-1,hidden_dim,T)
        for idx in range(len(self.obj_stem)):
            src_obj, src_obj_mask = self.obj_stem[idx](src_obj,src_obj_mask,src_query, src_query_mask)
        src_obj=src_obj.view(-1,hidden_dim,T)
        # vid_stem transformer
        
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_obj,src_obj_mask,src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)


        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks
    
@register_backbone("ObjectTokenTransformer")
class ObjectTokenTransformerBackbone(ConvTransformerBackbone):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,object_dim=512,object_win_size=1,object_use_cross_model=False,
            **kwargs
    ):
        super(ObjectTokenTransformerBackbone,self).__init__(**kwargs)
        n_embd=kwargs['n_embd']
        arch=kwargs['arch']
        n_head=kwargs['n_head']
        attn_pdrop=kwargs['attn_pdrop']
        proj_pdrop=kwargs['proj_pdrop']
        path_pdrop=kwargs['path_pdrop']
        n_embd_ks=kwargs['n_embd_ks']
        with_ln=kwargs['with_ln']
        self.obj_embd = nn.ModuleList()
        self.obj_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = object_dim
                # out_channels=4*n_embd
            # elif idx==arch[0]-1:
            #     in_channels = 4*n_embd
                # out_channels=n_embd
            else:
                in_channels = n_embd
                # out_channels=4*n_embd
            self.obj_embd.append(MaskedConv1D(
                in_channels=in_channels, out_channels=n_embd, kernel_size=1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.obj_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.obj_embd_norm.append(nn.Identity())
        self.obj_stem=nn.ModuleList()
        for idx in range(arch[2]):
            self.obj_stem.append(ObjectCAonlyTransformerBlock(
                n_embd, n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=object_win_size,
                use_cross_modal=object_use_cross_model,
            )
            )

        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(ObjectPlTransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()
        
        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))
        #obj_embedding
        src_obj=rearrange(src_obj,"b t o c -> (b o) c t")
        src_obj_mask=rearrange(src_obj_mask,"b t o -> (b o) 1 t")
        for idx in range(len(self.obj_embd)):
            src_obj, src_obj_mask = self.obj_embd[idx](src_obj, src_obj_mask)
            src_obj = self.relu(self.obj_embd_norm[idx](src_obj))
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        #obj_stem with txt_stem transformer
        hidden_dim=src_query.shape[1]
        src_obj=src_obj.view(B,-1,hidden_dim,T)
        for idx in range(len(self.txt_stem)):
            src_obj, src_obj_mask = self.obj_stem[idx](src_obj,src_obj_mask,src_query, src_query_mask)
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)
            
        src_obj=src_obj.view(-1,hidden_dim,T)
        # vid_stem transformer
        
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_obj,src_obj_mask,src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)


        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks
    
@register_backbone("ObjectTokenBMDTransformer")
class ObjectTokenBMDTransformerBackbone(ConvTransformerBackbone):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,object_dim=512,object_win_size=1,object_use_cross_model=False,
            **kwargs
    ):
        super(ObjectTokenBMDTransformerBackbone,self).__init__(**kwargs)
        n_embd=kwargs['n_embd']
        arch=kwargs['arch']
        n_head=kwargs['n_head']
        attn_pdrop=kwargs['attn_pdrop']
        proj_pdrop=kwargs['proj_pdrop']
        path_pdrop=kwargs['path_pdrop']
        n_embd_ks=kwargs['n_embd_ks']
        with_ln=kwargs['with_ln']
        self.obj_embd = nn.ModuleList()
        self.obj_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = object_dim
                # out_channels=4*n_embd
            # elif idx==arch[0]-1:
            #     in_channels = 4*n_embd
                # out_channels=n_embd
            else:
                in_channels = n_embd
                # out_channels=4*n_embd
            self.obj_embd.append(MaskedConv1D(
                in_channels=in_channels, out_channels=n_embd, kernel_size=1,
                stride=1, padding=0, bias=(not with_ln)
            )
            )
            if with_ln:
                self.obj_embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.obj_embd_norm.append(nn.Identity())
        self.obj_stem=nn.ModuleList()
        for idx in range(arch[2]):
            self.obj_stem.append(ObjectCAonlyTransformerBlock(
                n_embd, n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=object_win_size,
                use_cross_modal=object_use_cross_model,
            )
            )

        self.vid_stem = nn.ModuleList()
        for idx in range(arch[2]):
            self.vid_stem.append(ObjectBMDTransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe,
                use_cross_modal=True,
            )
            )
    def forward(self, src_vid, src_vid_mask, src_txt, src_txt_mask,src_obj,src_obj_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = src_vid.size()
        
        # vid_embedding network
        for idx in range(len(self.vid_embd)):
            src_vid, src_vid_mask = self.vid_embd[idx](src_vid, src_vid_mask)
            src_vid = self.relu(self.vid_embd_norm[idx](src_vid))
        #obj_embedding
        src_obj=rearrange(src_obj,"b t o c -> (b o) c t")
        src_obj_mask=rearrange(src_obj_mask,"b t o -> (b o) 1 t")
        for idx in range(len(self.obj_embd)):
            src_obj, src_obj_mask = self.obj_embd[idx](src_obj, src_obj_mask)
            src_obj = self.relu(self.obj_embd_norm[idx](src_obj))
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        assert src_txt is not None

        # txt_embedding network
        for idx in range(len(self.txt_embd)):
            src_txt, src_txt_mask = self.txt_embd[idx](src_txt, src_txt_mask)
            src_txt = self.relu(self.txt_embd_norm[idx](src_txt))

        src_query = src_txt
        src_query_mask = src_txt_mask

        #obj_stem with txt_stem transformer
        hidden_dim=src_query.shape[1]
        src_obj=src_obj.view(B,-1,hidden_dim,T)
        for idx in range(len(self.txt_stem)):
            src_obj, src_obj_mask = self.obj_stem[idx](src_obj,src_obj_mask,src_query, src_query_mask)
            src_query, src_query_mask = self.txt_stem[idx](src_query, src_query_mask)

        O=src_obj.shape[1]
        temp_obj_mask=src_obj_mask.view(B,O,1,T)
        src_obj_sum=src_obj.sum(dim=1)
        src_obj_num=torch.max(torch.tensor(1),temp_obj_mask.sum(dim=1))
        src_obj_mean=src_obj_sum/src_obj_num    

        # vid_stem transformer
        
        for idx in range(len(self.vid_stem)):
            src_vid, src_vid_mask = self.vid_stem[idx](src_vid, src_vid_mask, src_obj_mean,src_obj_mask,src_query, src_query_mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)


        
        for idx in range(len(self.branch)):
            src_vid, src_vid_mask = self.branch[idx](src_vid, src_vid_mask, src_query, src_query_mask)
            out_feats += (src_vid,)
            out_masks += (src_vid_mask,)

        return out_feats, out_masks
    