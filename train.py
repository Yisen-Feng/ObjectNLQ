# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.utils.data
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
from torch import nn
# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, save_checkpoint, make_optimizer, ReferringRecall,make_scheduler, fix_random_seed, ModelEma)
from libs.utils.train_utils import valid_one_epoch_loss,valid_one_epoch_nlq_singlegpu,valid_one_epoch_nlq_singlegpu_each_layer
from libs.utils.model_utils import count_parameters


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    init_process_group(backend="nccl")
    #关闭tokenizer并行化
    os.environ["TOKENIZERS_PARALLELISM"] = "false"#程序会自动把tokenizer作为加载数据中的一环，导致并行出错

    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')

    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if int(os.environ["LOCAL_RANK"]) == 0:
        pprint(cfg)
        os.makedirs(ckpt_folder, exist_ok=True)

    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= torch.cuda.device_count()
    # cfg['loader']['num_workers'] *= torch.cuda.device_count()
    print(cfg['opt']["learning_rate"])

    """2. create dataset / dataloader"""

    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    # val_loader = make_data_loader(
    #     val_dataset, False, None, **cfg['loader']
    # )
    val_loader = make_data_loader(
        val_dataset, False, rng_generator, **cfg['loader']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])#

    if int(os.environ["LOCAL_RANK"]) == 0:
        print(model)
        count_parameters(model)

    # enable model EMA
    # print("Using model EMA ...")
    model_ema = ModelEma(model)

    gpu_id = int(os.environ["LOCAL_RANK"])
    model = model.to(gpu_id)
    # model = DDP(model, device_ids=[gpu_id])

    if model_ema is not None:
        model_ema = model_ema.to(gpu_id)

    # optimizer
    if cfg['opt']["backbone_lr_weight"] == 1:
        optimizer = make_optimizer(model, cfg['opt'])
    else:
        optimizer = make_optimizer(model, cfg['opt'], head_backbone_group=True)
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        print(args.resume)
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(gpu_id))
            if 'state_dict_ema' in checkpoint.keys():
                pretrained_dict = checkpoint['state_dict_ema']
            else:
                pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict, strict=False)
            if args.resume_from_pretrain:
                args.start_epoch = 0
                for idx in range(len(model.backbone.vid_stem)):
                    if hasattr(model.backbone.vid_stem[idx], 'obj_attn') :
                        ori_module_name='backbone.vid_stem.{}.obj_attn.'.format(idx)
                        exist_module={k[len(ori_module_name):]: v for k, v in pretrained_dict.items() if k.startswith(ori_module_name)}
                        if len(exist_module)>0:
                            break

                        module_name='backbone.vid_stem.{}.cross_attn.'.format(idx)
                        backbone_vid_stem_cross_attn_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                        model.backbone.vid_stem[idx].obj_attn.load_state_dict(backbone_vid_stem_cross_attn_state_dict)

                        module_name='backbone.vid_stem.{}.ln3.'.format(idx)
                        backbone_vid_stem_ln3_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                        model.backbone.vid_stem[idx].ln_obj.load_state_dict(backbone_vid_stem_ln3_state_dict)
                    if hasattr(model.backbone.vid_stem[idx], 'obj_mlp') :
                        ori_module_name='backbone.vid_stem.{}.obj_mlp.'.format(idx)
                        exist_module={k[len(ori_module_name):]: v for k, v in pretrained_dict.items() if k.startswith(ori_module_name)}
                        if len(exist_module)>0:
                            break

                        module_name='backbone.vid_stem.{}.mlp.'.format(idx)
                        backbone_vid_stem_obj_mlp_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                        model.backbone.vid_stem[idx].obj_mlp.load_state_dict(backbone_vid_stem_obj_mlp_state_dict)

                        module_name='backbone.vid_stem.{}.ln2.'.format(idx)
                        backbone_vid_stem_ln2_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                        model.backbone.vid_stem[idx].ln_obj_mlp.load_state_dict(backbone_vid_stem_ln2_state_dict)
                if not isinstance(model.backbone.txt_embd, nn.Linear):
                    if hasattr(model.backbone, 'obj_embd'):
                        for idx in range(len(model.backbone.txt_embd)):
                        
                            ori_module_name='backbone.obj_embd.{}.'.format(idx)
                            exist_module={k[len(ori_module_name):]: v for k, v in pretrained_dict.items() if k.startswith(ori_module_name)}
                            if len(exist_module)>0:
                                break

                            module_name='backbone.txt_embd.{}.'.format(idx)
                            backbone_obj_embd_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                            model.backbone.obj_embd[idx].load_state_dict(backbone_obj_embd_state_dict)

                            module_name='backbone.txt_embd_norm.{}.'.format(idx)
                            backbone_obj_embd_norm_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                            model.backbone.obj_embd_norm[idx].load_state_dict(backbone_obj_embd_norm_state_dict)
                    if hasattr(model.backbone, 'obj_stem'):
                        ori_module_name='backbone.obj_stem.'.format(idx)
                        exist_module={k[len(ori_module_name):]: v for k, v in pretrained_dict.items() if k.startswith(ori_module_name)}
                        if len(exist_module)<=0:
                                
                            for idx in range(len(model.backbone.obj_stem)):
                                module_name='backbone.txt_stem.{}.attn.'.format(idx)
                                backbone_txt_stem_attn_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                                model.backbone.obj_stem[idx].cross_attn.load_state_dict(backbone_txt_stem_attn_state_dict)

                                module_name='backbone.txt_stem.{}.ln2.'.format(idx)
                                backbone_txt_stem_ln2_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                                model.backbone.obj_stem[idx].ln2.load_state_dict(backbone_txt_stem_ln2_state_dict)

                                module_name='backbone.txt_stem.{}.mlp.'.format(idx)
                                backbone_txt_stem_mlp_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                                model.backbone.obj_stem[idx].mlp.load_state_dict(backbone_txt_stem_mlp_state_dict)

                                module_name='backbone.txt_stem.{}.ln1.'.format(idx)
                                backbone_txt_stem_ln1_state_dict = {k[len(module_name):]: v for k, v in pretrained_dict.items() if k.startswith(module_name)}
                                model.backbone.obj_stem[idx].ln4.load_state_dict(backbone_txt_stem_ln1_state_dict)

                                model.backbone.obj_stem[idx].ln3.weigt=backbone_txt_stem_ln1_state_dict['weight'].unsqueeze(1)
                                model.backbone.obj_stem[idx].ln3.bias=torch.nn.Parameter(backbone_txt_stem_ln1_state_dict['bias'].unsqueeze(1))
            else:
                args.start_epoch = checkpoint['epoch'] + 1
                try:
                    model_ema.load_state_dict(checkpoint['state_dict_ema'])
                except:
                    pass
                # also load the optimizer / scheduler if necessary
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d})".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    elif args.resume_from_checkpoint:
        checkpoint=torch.load(args.resume_from_checkpoint, map_location=lambda storage, loc: storage.cuda(gpu_id))
        pretrained_dict = checkpoint['state_dict']
        
        # model.first_stage.load_state_dict([("first_stage."+key,pretrained_dict[key]) for key in pretrained_dict])
        model.first_stage.load_state_dict(pretrained_dict,strict=False)
        args.start_epoch = 0
        print("=> loaded checkpoint '{:s}' (epoch {:d})".format(
                args.resume_from_checkpoint, checkpoint['epoch']
            ))
        del checkpoint
    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    score_writer = open(os.path.join(ckpt_folder, "eval_results.txt"), mode="w", encoding="utf-8")
    mode=args.mode
    best_avgiou=0
    if hasattr(model,'max_epoch'):
        model.max_epoch=max_epochs
    
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_loader.sampler.set_epoch(epoch)
        # with torch.autograd.set_detect_anomaly(True):
        if mode not in ["eval","visualize"]:
            det_eval = ReferringRecall(dataset=cfg["track"],gt_file=cfg["dataset"]["json_file"])
            best_avgiou=train_one_epoch(
                    train_loader,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    model_ema=model_ema,
                    clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
                    tb_writer=tb_writer,
                    print_freq=args.print_freq,
                    mode=mode,
                    test_num=cfg['test_cfg'].get('test_num',1),
                    test_start_epoch=cfg['test_cfg'].get('test_start_epoch',0),
                    val_loader=val_loader,
                    det_eval=det_eval,
                    score_writer=score_writer,
                    best_avgiou=best_avgiou,
                    ckpt_folder=ckpt_folder
                )

            # save ckpt once in a while
            #交换了一下评测模型和保存模型的位置
            # if int(os.environ["LOCAL_RANK"]) == 0:
            #     save_states = {'epoch': epoch,
            #                 'state_dict': model.state_dict(),
            #                 'scheduler': scheduler.state_dict(),
            #                 'optimizer': optimizer.state_dict(),
            #                 'state_dict_ema': model_ema.module.state_dict(),
            #                 }

            #     save_checkpoint(
            #         save_states,
            #         False,
            #         file_folder=ckpt_folder,
            #         file_name='epoch_{:03d}.pth.tar'.format(epoch)
            #     )
        if (
                (epoch == max_epochs - 1) or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0)
                )
        ):
            print("\nStart testing model {:s} ...".format(cfg['model_name']))
            start = time.time()
            if mode not in ["eval","visualize"]:
                losses_tracker = valid_one_epoch_loss(
                    val_loader,
                    model,
                    epoch,
                    tb_writer=tb_writer,
                    print_freq=args.print_freq / 2,
                    mode=mode
                )
            det_eval = ReferringRecall(dataset=cfg["track"],gt_file=cfg["dataset"]["json_file"])
            if args.eval_each_layer:
                performance, score_strs,avgiou = valid_one_epoch_nlq_singlegpu_each_layer(
                    val_loader,
                    model,
                    -1,
                    evaluator=det_eval,
                    output_file=None,
                    tb_writer=None,
                    mode=mode,
                    layer_num=cfg['train_cfg']["num_decoder_layer"]
                )
            else:
                performance, score_strs,avgiou = valid_one_epoch_nlq_singlegpu(
                    val_loader,
                    model,
                    -1,
                    evaluator=det_eval,
                    output_file=None,
                    tb_writer=None,
                    mode=mode,
                    model_ema=model_ema
                )
            if avgiou>best_avgiou:
                best_avgiou=avgiou
            if int(os.environ["LOCAL_RANK"]) == 0:
                save_states = {'epoch': epoch,
                            # 'state_dict': model.state_dict(),
                            'state_dict': model_ema.module.state_dict(),
                            # 'scheduler': scheduler.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            # 'state_dict_ema': model_ema.module.state_dict(),
                            }
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='model_{}_{}.pth.tar'.format(epoch,avgiou)
                    # file_name='best_model.pth.tar'
                    # file_name='epoch_{:03d}_{:5f}.pth.tar'.format(epoch,best_avgiou)
                )
        
            end = time.time()
            print("All done! Total time: {:0.2f} sec".format(end - start))
            # print("losses_tracker: ", losses_tracker)
            score_str = "epoch{:d}\n".format(epoch)

            for key, value in losses_tracker.items():
                score_str += '\t{:s} {:.2f} ({:.2f})\n'.format(
                    key, value.val, value.avg
                )
            if int(os.environ["LOCAL_RANK"]) == 0:
                score_writer.write(score_strs+"avgiou={:04f}\n".format(avgiou))
                score_writer.write(score_str)
                score_writer.flush()

        

    # wrap up
    tb_writer.close()
    if int(os.environ["LOCAL_RANK"]) == 0:
        destroy_process_group()


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=1, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='./ckpt', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--resume_from_checkpoint', default='', type=str, metavar='PATH',
                        help='path to a pretrained checkpoint (default: none)')
    parser.add_argument('--resume_from_pretrain', default=False, type=bool)
    parser.add_argument('--eval_each_layer', default=False, type=bool)
    parser.add_argument("--mode", default="train", type=str, help="train or eval or debug or visualize")
    args = parser.parse_args()
    main(args)
