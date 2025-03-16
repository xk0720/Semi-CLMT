import random
import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from utils import ramps
from Data import dataloaders_glas
from monai.metrics.meandice import DiceMetric
from monai.losses.dice import DiceCELoss
from monai.metrics.utils import do_metric_reduction
from monai.networks import one_hot
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard-logs/log_1')
from networks_v2.new_our_net_distill_352 import Our_Net_distill as net_distill
from networks_v2.new_our_net_distill_352_ema import Our_Net_distill as net_distill_ema
from Models.unet_model import UNet
from monai.metrics.meaniou import MeanIoU
import torch.nn.functional as F
from Metrics.asc_loss import ASC_loss
import logging
from Models.ema import EMA_teacher
from datetime import datetime
from sklearn.metrics import f1_score


def get_log_path():
    current_time = datetime.now()
    time_str = str(current_time)
    time_str = '-'.join(time_str.split(' '))
    time_str = time_str.split('.')[0]
    log_path = "results/glas/log_" + time_str + '.txt'
    return log_path


def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(levelname)s: %(message)s'
    # DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    fhlr = logging.FileHandler(log_path)
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1.0 * ramps.sigmoid_rampup(epoch, 80)


def plot_sim_matrix(similarity_matrix, epoch):
    fig, ax = plt.subplots()
    # Plot the similarity matrix as a heatmap
    im = ax.imshow(similarity_matrix, cmap='Blues', vmin=0, vmax=1)
    # Set the axis labels and title
    ax.set_xticks(np.arange(0, 20, 5))
    ax.set_yticks(np.arange(0, 20, 5))
    cbar = ax.figure.colorbar(im, ax=ax)
    # Display the plot
    # plt.show()
    # Save as a png file
    plt.savefig('plots/glas/epoch_[{}]_myplot.png'.format(epoch))


def iou_similarity(score, target):
    smooth = 1e-6
    dim_len = len(score.size())

    # reducing only spatial dimensions (not batch nor channels)
    if dim_len == 5:
        dim = (3, 4)
    elif dim_len == 4:
        dim = (3,)

    intersect = torch.sum(score * target, dim=dim)  # intersect (N, N, Class)
    target_o = torch.sum(target, dim=dim)  # target_o (1, N, Class)
    score_o = torch.sum(score, dim=dim)  # score_o (N, 1, Class)
    union = target_o + score_o - intersect  # (N, N, Class)

    similarity = torch.where(union > 0, (intersect) / union, torch.tensor(1.0, device=target_o.device))
    t_zero = torch.zeros(1, device=similarity.device, dtype=similarity.dtype)
    nans = torch.isnan(similarity)
    not_nans = (~nans).float()
    not_nans = not_nans.sum(dim=-1)  # channel average, (dimension -1)

    dice_sim = torch.where(not_nans > 0, similarity.sum(dim=-1) / not_nans, t_zero)  # (N, N)

    return dice_sim


def save_checkpoint(state, ratio=1.0,
                    save_path='checkpoints_rite/unet'):
    logging.info('Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no.
    best_model = state['best_model']  # bool
    model = state['model']  # model type
    best_score = state['max_dice']  # dice score

    if best_model:
        filename = save_path + '/' + \
                   'best_model_[{}]_[{:.4f}].pth.tar'.format(ratio, best_score)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    print("Iinitialing weights now")
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def deapply_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()


def train_epoch(args, scaler, model, ema_model, device, train_loader, optimizer, epoch, Dice_CE_monai_loss,
                dice_similarity_criteria):
    t = time.time()

    # turn on mode 'train'
    model.train()

    # activate dropout
    ema_model.t_model.apply(deapply_dropout)

    loss_accumulator = []
    loss_accumulator_sup = []
    loss_accumulator_consist = []
    loss_accumulator_contras = []

    labeled_bs = args.labeled_bs
    batch_size = args.batch_size
    unlabeled_bs = batch_size - labeled_bs

    if args.ratio == 1.0:
        for batch_idx, (img_weak, mask_weak, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            # with autocast():

            if args.backbone == 'unet':
                output = model(img_weak.cuda())
            else:
                output, _, _ = model(img_weak.cuda())

            mask_weak = (mask_weak > 0).float()
            mask_weak = mask_weak.to(device)

            supervised_loss = Dice_CE_monai_loss(input=output, target=mask_weak)

            """ 
            The whole loss, # do ablation study
            """
            loss = supervised_loss

            # debug: Function 'ConvolutionBackward0' returned nan values in its 1th output.
            # if batch_idx == 60:
            #     v_n = []
            #     v_v = []
            #     v_g = []
            #     for name, parameter in model.named_parameters():
            #         v_n.append(name)
            #         v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
            #         v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
            #     for i in range(len(v_n)):
            #         print('value %s: %.3e ~ %.3e' % (v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
            #         print('grad  %s: %.3e ~ %.3e' % (v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))

            """ 
            Back Propagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            """
            loss.backward()
            optimizer.step()

            """ 
            Update the teacher model
            """
            ema_model.update()
            ema_model.apply_shadow()

            loss_accumulator.append(loss.item())
            if batch_idx + 1 < len(train_loader):
                print(
                    "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * len(img_weak),
                        len(train_loader.dataset),
                        100.0 * (batch_idx + 1) / len(train_loader),
                        loss.item(),
                        time.time() - t,
                    ),
                    end="",
                )
            else:
                print(
                    "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * len(img_weak),
                        len(train_loader.dataset),
                        100.0 * (batch_idx + 1) / len(train_loader),
                        np.mean(loss_accumulator),
                        time.time() - t,
                    )
                )

        # add to the tensorboard-logs
        writer.add_scalar("Whole_loss/train", np.mean(loss_accumulator), epoch)

    else:
        # get the training data
        for batch_idx, (img_weak, mask_weak, img_strong, _) in enumerate(train_loader):

            # with autocast():
            # get the output
            images = torch.cat([img_weak, img_strong], dim=0)
            # output, _, _ = model(images.cuda())
            if args.backbone == 'unet':
                output = model(images.cuda())
            else:
                output, _, _ = model(images.cuda())

            mask_weak = (mask_weak > 0).float()
            mask_weak = mask_weak.to(device)
            supervised_loss = Dice_CE_monai_loss(input=output[:labeled_bs], target=mask_weak[:labeled_bs])

            dice_contrastive_loss = dice_similarity_criteria(torch.softmax(output[batch_size + labeled_bs:], dim=1),
                                                             torch.softmax(output[labeled_bs:batch_size].detach(),
                                                                           dim=1))

            strong_noise = img_strong[labeled_bs:] + torch.clamp(torch.randn_like(img_strong[labeled_bs:]) * 0.1, 0,
                                                                 0.2)

            if args.backbone == 'unet':
                uw_ce_output = model(strong_noise.cuda())
            else:
                uw_ce_output, _, _ = model(strong_noise.cuda())

            T_num = 4
            _, _, w, h = images.shape
            unlabeled_bs = batch_size - labeled_bs
            preds = torch.zeros([T_num, unlabeled_bs, 2, w, h]).cuda()  # class num equals 2
            for i in range(T_num):
                noise_inputs_un = img_strong[labeled_bs:] + \
                                  torch.clamp(torch.randn_like(
                                      img_strong[labeled_bs:]) * 0.1, 0, 0.2)
                with torch.no_grad():
                    if args.backbone == 'unet':
                        prediction = ema_model.t_model(noise_inputs_un.cuda())
                    else:
                        prediction, _, _ = ema_model.t_model(noise_inputs_un.cuda())
                    # prediction, _, _ = ema_model.t_model(noise_inputs_un.cuda())
                    preds[i] = torch.softmax(prediction, dim=1)

            preds = torch.mean(preds, dim=0)
            masks_pred_mode2_certainty = preds

            weight = masks_pred_mode2_certainty.max(1)[0]
            ema_model.t_model.apply(deapply_dropout)  # not apply drop out

            with torch.no_grad():
                if args.backbone == 'unet':
                    output_t = ema_model.t_model((img_weak[labeled_bs:] +
                                                    torch.clamp(torch.randn_like(img_weak[labeled_bs:]) * 0.1, 0,
                                                                0.2)).cuda())
                else:
                    output_t, _, _ = ema_model.t_model((img_weak[labeled_bs:] +
                                                    torch.clamp(torch.randn_like(img_weak[labeled_bs:]) * 0.1, 0,
                                                                0.2)).cuda())

            # positive_loss_mat = F.cross_entropy(uw_ce_output, torch.argmax(output_t, dim=1), reduction="none")
            positive_loss_mat = F.cross_entropy(uw_ce_output, torch.softmax(output_t, dim=1), reduction="none")

            positive_loss_mat = positive_loss_mat * weight

            uncertainty_score = -1.0 * torch.sum(masks_pred_mode2_certainty * torch.log(masks_pred_mode2_certainty),
                                                 dim=1, keepdim=False)

            threshold = 0.5 + (np.log(2) - 0.5) * ramps.sigmoid_rampup(epoch, 150)

            uncertainty_mask = (uncertainty_score < threshold)  # uncertainty map

            cross_entropy_dist = torch.sum(positive_loss_mat[uncertainty_mask]) / (
                    2 * torch.sum(uncertainty_mask) + 1e-16)

            if args.rampup == 0:
                consistency_loss = 1.0 * cross_entropy_dist
            else:
                consistency_weight = get_current_consistency_weight(epoch)
                consistency_loss = consistency_weight * cross_entropy_dist

            loss = supervised_loss + consistency_loss + 0.5 * dice_contrastive_loss

            """ 
            Back Propagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ 
            Update the teacher model
            """
            ema_model.update()
            ema_model.apply_shadow()

            loss_accumulator.append(loss.item())
            loss_accumulator_sup.append(supervised_loss.item())
            loss_accumulator_consist.append(consistency_loss.item())
            loss_accumulator_contras.append(0.5 * dice_contrastive_loss.item())

            if batch_idx + 1 < len(train_loader):
                print(
                    "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * len(img_weak),
                        len(train_loader.dataset),
                        100.0 * (batch_idx + 1) / len(train_loader),
                        loss.item(),
                        time.time() - t,
                    ),
                    end="",
                )
            else:
                print(
                    "\rTrain Epoch: {}\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                        epoch,
                        # (batch_idx + 1) * len(img_weak),
                        # len(train_loader.dataset),
                        # 100.0 * (batch_idx + 1) / len(train_loader),
                        np.mean(loss_accumulator),
                        time.time() - t,
                    )
                )

        # add to the tensorboard-logs
        tag_scalar_dict = {'Sup_loss': np.mean(loss_accumulator_sup), 'Consist_loss': np.mean(loss_accumulator_consist),
                           'Contras_loss': np.mean(loss_accumulator_contras), 'Whole_loss': np.mean(loss_accumulator)}
        writer.add_scalars(main_tag='Loss/train', tag_scalar_dict=tag_scalar_dict, global_step=epoch)

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(args, model, device, test_loader, epoch):
    model.eval()

    # dice metric from monai
    dice_metric = DiceMetric(include_background=False)
    iou_metric = MeanIoU(include_background=False)
    outputs_list = []
    label_list = []

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        if args.backbone == 'unet':
            output = model(data)
        else:
            output, _, _ = model(data)
        outputs_list.append(output.detach().to("cpu").float())
        label_list.append(target)

    # prediction
    predictions = torch.cat(outputs_list, dim=0)
    # label
    softmax_predictions = one_hot(labels=torch.argmax(torch.softmax(predictions, dim=1), dim=1, keepdim=True),
                                  num_classes=2)
    labels = torch.cat(label_list, dim=0)
    labels = (labels > 0).float() # shape: (B, 1, H, W)
    one_hot_labels = one_hot(labels=labels, num_classes=2)

    val_dice = dice_metric._compute_tensor(y_pred=softmax_predictions, y=one_hot_labels)
    val_dice, _ = do_metric_reduction(f=val_dice)

    val_mIoU = iou_metric._compute_tensor(y_pred=softmax_predictions, y=one_hot_labels)
    val_mIoU, _ = do_metric_reduction(f=val_mIoU)

    # add the test performance to tensorboard-logs
    writer.add_scalar("Dice_score/test", val_dice, epoch)
    writer.add_scalar("mIoU/test", val_mIoU, epoch)

    # calculate f1 score
    y_pred = torch.argmax(predictions, dim=1, keepdim=False).view(-1)
    y_true = labels.view(-1)

    y_pred = np.array(y_pred, dtype='int')
    y_true = np.array(y_true, dtype='int')
    y_true[y_true != 0] = 1 # make sure either 0 or 1, in y_true
    f1 = f1_score(y_true, y_pred) # calculate f1 score

    return val_dice, val_mIoU, f1


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataloader, _, val_dataloader = dataloaders_glas.get_dataloaders(batch_size=args.batch_size, semi_ratio=args.ratio)

    if args.backbone == 'unet':
        model = UNet(n_channels=3, n_classes=2, bilinear=False)
        teacher = UNet(n_channels=3, n_classes=2, bilinear=False)
    else:
        model = net_distill()  # no dropout (352 * 352)
        teacher = net_distill_ema()  # with dropout (352 * 352)

    model.to(device)
    teacher.to(device)

    # create EMA_teacher
    ema_model = EMA_teacher(s_model=model, t_model=teacher, decay=0.99)
    ema_model.register()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return (
        None,
        device,
        train_dataloader,
        val_dataloader,
        model,
        ema_model,
        optimizer,
    )


def train(args):
    (image_input,
     device,
     train_dataloader,
     val_dataloader,
     model,
     ema_model,
     optimizer,
     ) = build(args)

    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.8, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.8, verbose=True
            )

    # save evaluation results to a txt file
    log_path = get_log_path()
    set_logging(log_path=log_path)

    # max_dice is created for better saving the model weights
    max_dice_score = 0.0
    max_mIoU = 0.0
    max_dice = 0.0
    max_f1 = 0.0

    # set a gradscaler
    scaler = GradScaler()

    # for semi-supervised learning setting
    labeled_bs = args.labeled_bs
    batch_size = args.batch_size
    unlabeled_bs = batch_size - labeled_bs

    Dice_CE_monai_loss = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, reduction="mean")
    dice_similarity_criteria = ASC_loss(batch_size=unlabeled_bs, device=device)

    # construct new train_loader (dataloader), batch_size == len(dataset)
    # train_data_loader = dataloaders_RITE_visual.get_dataloaders(batch_size=5)

    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(args, scaler, model, ema_model, device, train_dataloader,
                               optimizer, epoch, Dice_CE_monai_loss, dice_similarity_criteria)

            val_dice, val_miou, f1_score = test(args, model, device, val_dataloader, epoch)
            # print(f1_score)

            if val_dice.item() > max_dice_score:
                max_dice_score = val_dice.item()
                max_mIoU = val_miou.item()
                max_f1 = f1_score

            logging.info("-" * 70)
            logging.info("Epoch: [{}], Training loss: [{:.4f}]".format(epoch, loss))
            logging.info("Epoch: [{}], Dice: [{:.4f}], mIoU: [{:.4f}], f1 score: [{:.4f}]".format(
                epoch, val_dice.item(), val_miou.item(), f1_score))
            logging.info("Epoch: [{}], Max Dice: [{:.4f}], Max mIoU: [{:.4f}], f1 score: [{:.4f}]".format(
                epoch, max_dice_score, max_mIoU, max_f1))

            if val_dice > max_dice:
                if epoch > 50:
                    logging.info('Saving best model!')
                    max_dice = val_dice  # update the highest dice score
                    # save the checkpoint
                    # save_checkpoint(state={'epoch': epoch,
                    #                        'best_model': True,
                    #                        'model': 'Swin_Unet_v2',
                    #                        'state_dict': model.state_dict(),
                    #                        'max_dice': max_dice.item(),
                    #                        'optimizer': optimizer.state_dict()}, ratio=args.ratio)

        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)

        # apply learning rate decay
        if args.lrs == "true":
            scheduler.step(val_dice)


    # shut down the tensorboard-logs
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dataset", type=str, default="Glas")
    parser.add_argument("--root", type=str, default="/root/autodl-tmp/myData_kvasir/Kvasir-SEG/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--labeled-bs", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=5e-5, dest="lr")
    parser.add_argument("--rampup", type=int, default=1)
    parser.add_argument("--visual", type=str, default='rite')
    parser.add_argument("--backbone", type=str, default='ours')
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )
    parser.add_argument(
        "--img_size", type=int, default="224"
    )
    parser.add_argument(
        "--num_classes", type=int, default="2"
    )

    return parser.parse_args()


def main():
    args = get_args()
    init_seed = args.seed
    print("init seed is: ", init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    train(args)


if __name__ == "__main__":
    main()
