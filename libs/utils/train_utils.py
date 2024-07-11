import os
import time

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR, WarmupLRScheduler
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm,LayerNorm4
from ..modeling.head import PerLengthRegHead
from transformers.models.t5.modeling_t5 import  T5LayerNorm
from basic_utils import save_json
import pkg_resources
def is_version_greater(package_name, target_version):
    current_version = pkg_resources.get_distribution(package_name).version
    return pkg_resources.parse_version(current_version) > pkg_resources.parse_version(target_version)


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = False
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
        # torch.use_deterministic_algorithms(True, warn_only=True)
        # torch.autograd.set_detect_anomaly(True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config, head_backbone_group=False):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    if head_backbone_group:
        head_decay = set()
        backbone_decay = set()
        head_no_decay = set()
        backbone_no_decay = set()

    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D,torch.nn.MultiheadAttention,torch.nn.Embedding)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm,torch.nn.LayerNorm,LayerNorm4,T5LayerNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            # elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath,PerLengthRegHead)):
            elif pn.endswith('scale') :
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)
            elif pn.endswith('mu') or pn.endswith('reg_left') or pn.endswith('reg_right') or pn.endswith('sigma'):
                no_decay.add(fpn)
            elif pn.endswith('A_log') or pn.endswith('D') or pn.endswith('A_b_log') or pn.endswith('D_b'):
                no_decay.add(fpn)


    if head_backbone_group:#false
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # if "head" in fpn:  # or "txt" in fpn
                if "head" in fpn or "decoder" in fpn:
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        head_no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        head_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        head_no_decay.add(fpn)
                    elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                        # corner case of our scale layer
                        head_no_decay.add(fpn)
                    elif pn.endswith('rel_pe'):
                        # corner case for relative position encoding
                        head_no_decay.add(fpn)
                else:
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        backbone_no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        backbone_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        backbone_no_decay.add(fpn)
                    elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                        # corner case of our scale layer
                        backbone_no_decay.add(fpn)
                    elif pn.endswith('rel_pe'):
                        # corner case for relative position encoding
                        backbone_no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params),)

    # print("decay: ", len(decay))
    # for pn in sorted(list(decay)):
    #     print(pn)
    #
    # print("no_decay: ", len(no_decay))
    # for pn in sorted(list(no_decay)):
    #     print(pn)

    if head_backbone_group:#false
        print("param_dict.keys(): ", len(param_dict.keys()))
        inter_params = head_decay & head_no_decay & backbone_decay & backbone_no_decay
        union_params = head_decay | head_no_decay | backbone_decay | backbone_no_decay
        inter_decay_params = head_decay & backbone_decay
        inter_no_decay_params = head_no_decay & backbone_no_decay
        inter_backbone_params = backbone_decay & backbone_no_decay
        # print("union_params: ", len(union_params))
        assert len(inter_decay_params) == 0, "parameters %s made it into both head_decay/backbone_decay sets!" \
                                             % (str(inter_decay_params),)
        assert len(inter_no_decay_params) == 0, "parameters %s made it into both head_no_decay/backbone_no_decay sets!" \
                                                % (str(inter_no_decay_params),)
        assert len(inter_backbone_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_backbone_params),)
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)

        print("decay: ", len(decay))
        print("head_decay: ", len(head_decay))
        print("backbone_decay: ", len(backbone_decay))
        print("no_decay: ", len(no_decay))
        print("head_no_decay: ", len(head_no_decay))
        print("backbone_no_decay: ", len(backbone_no_decay))

    # print("no_decay: ", len(no_decay))
    # for pn in sorted(list(no_decay)):
    #     print(pn)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    if head_backbone_group:
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(head_decay))],
             "weight_decay": optimizer_config['weight_decay']
                , 'lr': optimizer_config["learning_rate"]},
            {"params": [param_dict[pn] for pn in sorted(list(head_no_decay))], "weight_decay": 0.0
                , 'lr': optimizer_config["learning_rate"]},
            {"params": [param_dict[pn] for pn in sorted(list(backbone_decay))],
             "weight_decay": optimizer_config['weight_decay']
                , 'lr': optimizer_config["learning_rate"] * optimizer_config["backbone_lr_weight"], },
            {"params": [param_dict[pn] for pn in sorted(list(backbone_no_decay))], "weight_decay": 0.0
                , 'lr': optimizer_config["learning_rate"] * optimizer_config["backbone_lr_weight"], },
        ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
        optimizer,
        optimizer_config,
        num_iters_per_epoch,
        last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )
        elif optimizer_config["schedule_type"] == "constant":
            # Cosine
            scheduler = WarmupLRScheduler(
                optimizer,
                warmup_steps,
                last_epoch=last_epoch
            )
        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(
        train_loader,
        model,
        optimizer,
        scheduler,
        curr_epoch,
        model_ema=None,
        clip_grad_l2norm=-1,
        tb_writer=None,
        print_freq=20,
        mode="train",
        test_num=1,
        test_start_epoch=0,
        val_loader=None,
        det_eval=None,
        score_writer=None,
        best_avgiou=0,
        ckpt_folder=None
):
    """Training the model for one epoch"""
    # set up meters
    gpu_id = int(os.environ["LOCAL_RANK"])
    # model = model.to(gpu_id)
    # try:
    #     model = DDP(model, device_ids=[gpu_id])#find_unused_parameters=True
    # except:

    # number of iterations per epoch
    num_iters = len(train_loader)
    if torch.distributed.is_initialized():
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)  #
        train_loader.sampler.set_epoch(curr_epoch)
        if hasattr(model.module,'cur_epoch'):
            model.module.cur_epoch=curr_epoch
            model.module.max_iters=num_iters
    else:
        model = model.to(gpu_id)
        if hasattr(model,'cur_epoch'):
            model.cur_epoch=curr_epoch
            model.max_iters=num_iters
    #
    # if model_ema is not None:
    #     model_ema = model_ema.to(gpu_id)
    # model_ema = DDP(model_ema, device_ids=[gpu_id])

    batch_time = AverageMeter()
    losses_tracker = {}
    
    
    # switch to train mode
    model.train()
    if hasattr(model,'cur_epoch'):
        model.cur_epoch=curr_epoch
        model.max_iters=num_iters
    # main training loop
    print("\n[Train]: [GPU{:d}] Epoch {:d} started".format(gpu_id, curr_epoch))
    start = time.time()
    for iter_idx, video_list in tqdm.tqdm(enumerate(train_loader, 0), desc="training one epoch"):
        # model_inputs, targets \
        #     = prepare_batch_inputs(batch[1], model.device, non_blocking=True)
        # zero out optim
        if torch.distributed.is_initialized():
            if hasattr(model.module,'cur_iter'):
                model.module.cur_iter=iter_idx
        else:
            if hasattr(model,'cur_iter'):
                model.cur_iter=iter_idx
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        # with torch.autograd.detect_anomaly():
        losses = model(video_list)
        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)
        if mode=='debug':
            break
        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
        # if (iter_idx != 0) and (iter_idx % 10) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                if torch.distributed.is_initialized():
                    lr_tensor=torch.tensor(lr).to(gpu_id)#[2,3]
                    torch.distributed.all_reduce(lr_tensor, op=torch.distributed.ReduceOp.SUM)
                    world_size = torch.distributed.get_world_size()
                    lr_tensor=lr_tensor/world_size
                    lr=lr_tensor.to('cpu').numpy()
                if gpu_id==0:
                    tb_writer.add_scalar(
                        'train/GPU_{}/learning_rate'.format(gpu_id),
                        lr,
                        global_step
                    )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        if torch.distributed.is_initialized():
                            value_tensor=torch.tensor(value.val).to(gpu_id)#[2,3]
                            torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.SUM)
                            world_size = torch.distributed.get_world_size()
                            value_tensor=value_tensor/world_size
                            value=value_tensor.to('cpu').numpy()
                            tag_dict[key] = value
                        else:
                            tag_dict[key] = value.val
                if torch.distributed.is_initialized():
                    final_loss_tensor=torch.tensor(losses_tracker['final_loss'].val).to(gpu_id)#[2,3]
                    torch.distributed.all_reduce(final_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                    world_size = torch.distributed.get_world_size()
                    final_loss_tensor=final_loss_tensor/world_size
                    final_loss=final_loss_tensor.to('cpu').numpy()
                else:
                    final_loss=losses_tracker['final_loss'].val
                if gpu_id==0:
                    tb_writer.add_scalars(
                        'train/GPU_{}/all_losses'.format(gpu_id),
                        tag_dict,
                        global_step
                    )
                    # final loss
                    tb_writer.add_scalar(
                        'train/GPU_{}/final_loss'.format(gpu_id),
                        final_loss,
                        global_step
                    )

            # print to terminal
            block1 = '[GPU{:d}] Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                gpu_id, curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block3, block4]))

        #test in epoch

        if test_num>1:
            if curr_epoch>=test_start_epoch:
                if (iter_idx+1)%(num_iters//test_num+1)==0:
                    performance, score_strs,avgiou = valid_one_epoch_nlq_singlegpu(
                        val_loader,
                        model,
                        -1,
                        evaluator=det_eval,
                        output_file=None,
                        tb_writer=None,
                        mode=mode
                    )
                    if avgiou>best_avgiou:
                        best_avgiou=avgiou
                        if int(os.environ["LOCAL_RANK"]) == 0:
                                save_states = {'epoch': curr_epoch,
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
                                    # file_name='best_model_{}.pth.tar'.format(avgiou)
                                    file_name='best_model.pth.tar'
                                    # file_name='epoch_{:03d}_{:04d}_{:04f}.pth.tar'.format(curr_epoch,iter_idx,best_avgiou)
                                )
                    end = time.time()
                    print("All done! Total time: {:0.2f} sec".format(end - start))
                    # print("losses_tracker: ", losses_tracker)
                    score_str = "epoch{:d}item{:d}\n".format(curr_epoch,iter_idx)

                    # for key, value in losses_tracker.items():
                    #     score_str += '\t{:s} {:.2f} ({:.2f})\n'.format(
                    #         key, value.val, value.avg
                    #     )
                    if int(os.environ["LOCAL_RANK"]) == 0:
                        score_writer.write(score_strs+"avgiou={:04f}\n".format(avgiou))
                        score_writer.write(score_str)
                        score_writer.flush()
                    model.train()
                    start=time.time()
    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: [GPU{:d}] Epoch {:d} finished with lr={:.8f}\n".format(gpu_id, curr_epoch, lr))
    return best_avgiou


def valid_one_epoch_loss(
        eval_loader,
        model,
        curr_epoch,
        tb_writer=None,
        print_freq=20,
        mode="train"
):
    """Eval the model for one epoch"""
    # set up meters
    gpu_id = int(os.environ["LOCAL_RANK"])
    model = model.to(gpu_id)
    if torch.distributed.is_initialized():
        model = DDP(model, device_ids=[gpu_id])

    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(eval_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[EVAL]: [GPU{:d}] Epoch {:d} started".format(gpu_id, curr_epoch))
    start = time.time()
    for iter_idx, video_list in tqdm.tqdm(enumerate(eval_loader, 0), desc="validate one epoch"):
        with torch.no_grad():
            # forward the model
            losses = model(video_list)

            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()
            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())
            if mode=="debug":
                break
            # printing (only check the stats when necessary to avoid extra cost)
            if (iter_idx != 0) and (iter_idx % print_freq) == 0:
                # log to tensor board
                global_step = curr_epoch * num_iters + iter_idx
                if tb_writer is not None:
                    # all losses
                    tag_dict = {}
                    for key, value in losses_tracker.items():
                        if key != "final_loss":
                            if torch.distributed.is_initialized():
                                value_tensor=torch.tensor(value.val).to(gpu_id)#[2,3]
                                torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.SUM)
                                world_size = torch.distributed.get_world_size()
                                value_tensor=value_tensor/world_size
                                value=value_tensor.to('cpu').numpy()
                                tag_dict[key] = value
                            else:
                                tag_dict[key] = value.val
                    if torch.distributed.is_initialized():
                        final_loss_tensor=torch.tensor(losses_tracker['final_loss'].val).to(gpu_id)#[2,3]
                        torch.distributed.all_reduce(final_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                        world_size = torch.distributed.get_world_size()
                        final_loss_tensor=final_loss_tensor/world_size
                        final_loss=final_loss_tensor.to('cpu').numpy()
                    else:
                        final_loss=losses_tracker['final_loss'].val
                    if gpu_id==0:
                        tb_writer.add_scalars(
                            'EVAL/GPU_{}/all_losses'.format(gpu_id),
                            tag_dict,
                            global_step
                        )
                        # final loss
                        tb_writer.add_scalar(
                            'EVAL/GPU_{}/final_loss'.format(gpu_id),
                            final_loss,
                            global_step
                        )
                # print to terminal
                block1 = '[GPU{:d}] Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                    gpu_id, curr_epoch, iter_idx, num_iters
                )
                block2 = 'Time {:.2f} ({:.2f})'.format(
                    batch_time.val, batch_time.avg
                )
                block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                    losses_tracker['final_loss'].val,
                    losses_tracker['final_loss'].avg
                )
                block4 = ''
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                            key, value.val, value.avg
                        )

                print('\t'.join([block1, block2, block3, block4]))

    # finish up and print
    print("[EVAL]: [GPU{:d}] Epoch {:d} \n".format(gpu_id, curr_epoch))
    return losses_tracker

def save_result(results, output_file,dataset):
    save_submission = []
    for item in tqdm.tqdm(results):
        new_item = item.copy()
        new_item["predicted_times"] = new_item["predicted_times"][:10]
        save_submission.append(new_item)
    if dataset=='nlq':
        save_data = {
            "version": "1.0",
            "challenge": "ego4d_nlq_challenge",
            "results": save_submission,
        }
    elif dataset=='goal_step':
        save_data = {
            "version": "1.0",
            "challenge": "ego4d_goalstep_challenge",
            "results": save_submission,
        }
    save_json(save_data, output_file)
    return
def valid_one_epoch_nlq_singlegpu(
        val_loader,
        model,
        curr_epoch,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20,
        mode="train",
        model_ema=None
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)
    gpu_id = int(os.environ["LOCAL_RANK"])
    if model_ema is not None:
        model =model_ema.module
    model = model.to(gpu_id)
    if torch.distributed.is_initialized():
        model = DDP(model, device_ids=[gpu_id])

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = []
    results_noscore = []
    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            # for item in video_list:
            #     print(item)
            output = model(video_list,mode=mode)
            # print("output, ", output)
            # upack the results into ANet format
            num_vids = len(output)

            if evaluator.dataset in ["nlq","goal_step"]:
                for vid_idx in range(num_vids):
                    qid = video_list[vid_idx]['query_id']
                    temp_list = qid.split("_")
                    assert output[vid_idx]['segments'].shape[0] > 0
                    new_prediction = [segment + [score] for segment, score in
                                      zip(output[vid_idx]['segments'].cpu().detach().tolist(),
                                          output[vid_idx]['scores'].cpu().detach().tolist())]
                    
                    results.append({
                        'query_idx': int(temp_list[-1]),
                        'annotation_uid': "_".join(temp_list[:-1]),
                        'predicted_times': new_prediction,  # output[vid_idx]['segments'].cpu().detach().tolist(),
                        'clip_uid': video_list[vid_idx]['video_id'],
                        # 'score': output[vid_idx]['scores'].cpu().detach().tolist(),
                    })
                    if output_file is not None:
                        new_prediction_noscore = [segment for segment in
                                            output[vid_idx]['segments'].cpu().detach().tolist()]
                        results_noscore.append({
                            'query_idx': int(temp_list[-1]),
                            'annotation_uid': "_".join(temp_list[:-1]),
                            'predicted_times': new_prediction_noscore,  # output[vid_idx]['segments'].cpu().detach().tolist(),
                            'clip_uid': video_list[vid_idx]['video_id'],
                        })
            else:
                for vid_idx in range(num_vids):
                    assert output[vid_idx]['segments'].shape[0] > 0
                    new_prediction = [segment + [score] for segment, score in
                                      zip(output[vid_idx]['segments'].cpu().detach().tolist(),
                                          output[vid_idx]['scores'].cpu().detach().tolist())]
                    results.append({
                        'query_id': video_list[vid_idx]['query_id'],
                        'predicted_times': new_prediction,
                        'video_id': video_list[vid_idx]['video_id'],
                    })
            if mode=="debug":
                break
        # printing
        if (iter_idx != 0) and iter_idx % print_freq == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    if output_file is not None:
        # dump to a pickle file that can be directly used for evaluation
        # with open(output_file, "wb") as f:
        save_result(results, output_file[0],evaluator.dataset)
        save_result(results_noscore, output_file[1],evaluator.dataset)
        return results
    # assert evaluator.dataset == "ego4d"
    performance, score_str = evaluator.evaluate(results, verbose=True)
    if torch.distributed.is_initialized():
        performance_tensor=torch.tensor(performance).to(gpu_id)#[2,3]
        torch.distributed.all_reduce(performance_tensor, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        performance_tensor=performance_tensor/world_size
        performance=performance_tensor.to('cpu').numpy()
    if evaluator.dataset=='nlq':
            metric=performance[:,0].mean()#R@1
    elif evaluator.dataset=="goal_step":
            metric=performance[0,0]#R@1，iou@0.3
    # performance=performance/100
    score_str = evaluator.display_results(performance)
    print(score_str, flush=True)
    # log mAP to tb_writer
    # if tb_writer is not None:
    #     tb_writer.add_scalar('validation/mAP', performance, curr_epoch)

    return performance, score_str,metric
def valid_one_epoch_nlq_singlegpu_each_layer(
        val_loader,
        model,
        curr_epoch,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20,
        mode="train",
        layer_num=6
):
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    # results = [[]]*(layer_num+1)#大错特错，这样写是浅拷贝，导致每个元素都是同一个list
    results = [[] for _ in range(layer_num+2)]

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            # for item in video_list:
            #     print(item)
            output = model(video_list,visualize=(mode=="visualize"))
            # print("output, ", output)
            # upack the results into ANet format
            
            if evaluator.dataset in ["nlq","goal_step"]:
                for i,output_layer in enumerate(output):
                    num_vids = len(output_layer)
                    for vid_idx in range(num_vids):
                        qid = video_list[vid_idx]['query_id']
                        temp_list = qid.split("_")
                        assert output_layer[vid_idx]['segments'].shape[0] > 0
                        new_prediction = [segment + [score] for segment, score in
                                        zip(output_layer[vid_idx]['segments'].cpu().detach().tolist(),
                                            output_layer[vid_idx]['scores'].cpu().detach().tolist())]
                        results[i].append({
                            'query_idx': int(temp_list[-1]),
                            'annotation_uid': "_".join(temp_list[:-1]),
                            'predicted_times': new_prediction,  # output[vid_idx]['segments'].cpu().detach().tolist(),
                            'clip_uid': video_list[vid_idx]['video_id'],
                            # 'score': output[vid_idx]['scores'].cpu().detach().tolist(),
                        })
            else:
                for vid_idx in range(num_vids):
                    assert output[vid_idx]['segments'].shape[0] > 0
                    new_prediction = [segment + [score] for segment, score in
                                      zip(output[vid_idx]['segments'].cpu().detach().tolist(),
                                          output[vid_idx]['scores'].cpu().detach().tolist())]
                    results.append({
                        'query_id': video_list[vid_idx]['query_id'],
                        'predicted_times': new_prediction,
                        'video_id': video_list[vid_idx]['video_id'],
                    })
            if mode in "debug":
                break
        # printing
        if (iter_idx != 0) and iter_idx % print_freq == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    if output_file is not None:
        # dump to a pickle file that can be directly used for evaluation
        # with open(output_file, "wb") as f:
        save_submission = []
        for item in tqdm.tqdm(results):
            new_item = item.copy()
            new_item["predicted_times"] = new_item["predicted_times"][:10]
            save_submission.append(new_item)

        save_data = {
            "version": "1.0",
            "challenge": "ego4d_nlq_challenge",
            "results": save_submission,
        }
        save_json(save_data, output_file)

    # assert evaluator.dataset == "ego4d"
    score_strs=""
    for i,result in enumerate(results):
        if len(result)==0:
            break
        score_strs+="layer{}\n".format(i)
        performance, score_str,metric = evaluator.evaluate(result, verbose=True)
        score_strs+=score_str
        score_strs+="\n"
    # log mAP to tb_writer
    # if tb_writer is not None:
    #     tb_writer.add_scalar('validation/mAP', performance, curr_epoch)
    
    return performance, score_strs,avgiou