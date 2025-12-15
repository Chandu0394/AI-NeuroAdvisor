import warnings
warnings.filterwarnings("ignore")
import logging
import os
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from setproctitle import setproctitle
import torch.nn.functional as F
from torch.utils.data import DataLoader



from utils.parser import ParserUse

from model.LSD_transformer import LSD_Transformer
from dataset.CNCSP import VideoSample, FeatureDataset
from utils.util import plot_loss
from utils.losses import FocalLoss
from xlstm1.xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

def train_trans(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info("\n\n\n" + "|| "*10 + "Begin training transformer")
    setproctitle("Trans")

    transformer = LSD_Transformer(args.mstcn_f_maps, args.mstcn_f_dim, args.out_classes, args.trans_seq,
                              d_model=args.mstcn_f_maps)

    if os.path.isfile(args.trans_iter):
        paras = torch.load(args.trans_iter)
        transformer.load_state_dict(paras)
    transformer.cuda()


    # 标记
    focal_loss = FocalLoss(weight=torch.tensor([2.0,3.0,1.0,1.0,7.0,1.0,13.0]).cuda())
    focal_loss.cuda()
    with open(args.data_file, "rb") as f:
        data_dict = pickle.load(f)
    with open(args.emb_file, "rb") as f:
        emb_dict = pickle.load(f)

    train_data = VideoSample(data_dict=data_dict, data_idxs=args.train_names, data_features=emb_dict, is_train=True)
    val_data = VideoSample(data_dict=data_dict, data_idxs=args.val_names, data_features=emb_dict, is_train=True)
    train_loader = DataLoader(train_data, batch_size=args.trans_bs, num_workers=8, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.trans_bs, num_workers=8, drop_last=True, shuffle=False)

    iterations = 1
    train_losses = []
    val_losses = []
    while iterations < args.trans_iterations:
        for data in train_loader:

            # fusion_model1.train()
            transformer.train()
            img_featrues0, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)

            cfg = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=1
                    )
                ),
                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend="cuda",
                        num_heads=1,
                        conv1d_kernel_size=4,
                        bias_init="powerlaw_blockdependent",
                    ),

                    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
                ),
                context_length=2048,
                num_blocks=2,
                embedding_dim=img_featrues0.size(1),
                slstm_at=[],

            )

            fusion_model = xLSTMBlockStack(cfg)

            fusion_model.cuda()

            fusion_model.train()

            optimizer = optim.SGD([{"params": transformer.parameters()}, {"params": fusion_model.parameters()}],
                                  lr=args.trans_lr, weight_decay=args.trans_weight_decay, momentum=0.9)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.trans_steps, gamma=0.1)
            img_featrues = torch.transpose(img_featrues0, 1, 2)
            features = fusion_model(img_featrues).squeeze(1)



            p_classes = transformer(features, img_featrues0).squeeze()
            p_loss = focal_loss(p_classes, labels.squeeze())  # weight=weights)

            optimizer.zero_grad()
            p_loss.backward()
            optimizer.step()

            if iterations % 50 == 0:
                logging.info("Iterations {:>10d} / {}, loss {:>10.5f}".format(iterations, args.trans_iterations, p_loss.item()))
                train_losses.append([iterations, p_loss.item()])

            if iterations % 100 == 0:


                transformer.eval()
                with torch.no_grad():
                    val_loss = []
                    for data in val_loader:
                        img_featrues0, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)


                        cfg = xLSTMBlockStackConfig(
                            mlstm_block=mLSTMBlockConfig(
                                mlstm=mLSTMLayerConfig(
                                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=1
                                )
                            ),
                            slstm_block=sLSTMBlockConfig(
                                slstm=sLSTMLayerConfig(
                                    backend="cuda",
                                    num_heads=1,
                                    conv1d_kernel_size=4,
                                    bias_init="powerlaw_blockdependent",
                                ),
                                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
                            ),
                            context_length=2048,
                            num_blocks=2,
                            embedding_dim=img_featrues0.size(1),
                            slstm_at=[],

                        )

                        fusion_model = xLSTMBlockStack(cfg)
                        fusion_model.cuda()
                        fusion_model.eval()


                        img_featrues = torch.transpose(img_featrues0, 1, 2)
                        features = fusion_model(img_featrues).squeeze(1)

                        p_classes = transformer(features, img_featrues0).squeeze()
                        p_loss = F.cross_entropy(p_classes, labels.squeeze())  # , weight=weights)
                        val_loss.append(p_loss.item())
                    mean_loss = sum(val_loss) / len(val_loss)
                    logging.info(">> "*10 + "Validation loss at iteration {:>10d} is {:>10.5f}".format(iterations, sum(val_loss) / len(val_loss)))
                    val_losses.append([iterations, mean_loss])
                    plot_loss(train_losses, val_losses, "<path>".format(args.log_time))

            iterations += 1
            lr_scheduler.step()
            if iterations > args.trans_iterations:
                break

    save_file = os.path.join(args.save_model, f"fusion_{args.log_time}.pth")
    args.fusion_model = save_file
    torch.save(fusion_model.state_dict(), save_file)

    save_file = os.path.join(args.save_model, f"transformer_{args.log_time}.pth")
    args.trans_model = save_file
    torch.save(transformer.state_dict(), save_file)
    logging.info(f"Trained transformer saved to {save_file}")

    return args

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--cfg", default="train", type=str, required=True)
    args.add_argument("-n", default="", type=str, help="Notes for training")
    args = args.parse_args()

    args = ParserUse(args.cfg, "transformer").add_args(args)

    logging.info(args)
    logging.info("==" * 10)
    train_trans(args)

