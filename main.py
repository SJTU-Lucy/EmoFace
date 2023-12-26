import re, random, math
import numpy as np
import argparse
import tensorboardX

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os, shutil
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import getDataLoader, FeatureDataset, FeaturesConstructor, AudioDataProcessor
from net import EmoFace


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss().to(device)
MAE = nn.L1Loss().to(device)

def trainer(args, train_loader, dev_loader, model, optimizer, scheduler, epoch, writer):
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    iteration = 0
    for e in range(0, epoch + 1):
        mse_log = []
        mae_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (data, target) in pbar:
            iteration += 1
            # to gpu
            audio = data[0].to(torch.float32).to(device)
            label = data[1].to(torch.float32).to(device)
            emotion = data[2].to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)
            output = model(audio, label, emotion)
            mse = MSE(output, target)
            mae = MAE(output, target)
            mse.backward()
            mse_log.append(mse.item())
            mae_log.append(mae.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            writer.add_scalar("train_mse", mse.item(), iteration)
            writer.add_scalar("train_mae", mae.item(), iteration)
            pbar.set_description("(Epoch {}, iteration {}) MSE LOSS:{:.7f} MAE LOSS:{:.7f}".
                                 format((e + 1), iteration, np.mean(mse_log), np.mean(mae_log)))
        writer.add_scalar("train_avg_mse", np.mean(mse_log), e)
        writer.add_scalar("train_avg_mae", np.mean(mae_log), e)

        # validation
        valid_mse_log = []
        valid_mae_log = []
        model.eval()
        for data, target in dev_loader:
            # to gpu
            audio = data[0].to(torch.float32).to(device)
            label = data[1].to(torch.float32).to(device)
            emotion = data[2].to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)
            output = model(audio, label, emotion)
            mse = MSE(output, target)
            mae = MAE(output, target)
            valid_mse_log.append(mse.item())
            valid_mae_log.append(mae.item())

        writer.add_scalar("valid_avg_mse", np.mean(valid_mse_log), e)
        writer.add_scalar("valid_avg_mae", np.mean(valid_mae_log), e)
        print("epcoh: {}, MSE loss:{:.7f} MAE loss:{:.7f}".format(e + 1, np.mean(valid_mse_log),
                                                                  np.mean(valid_mae_log)))

        # schedule
        scheduler.step()

        # save
        if (e > 0 and e % 200 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='EmoFace: Audio-driven Emotional 3D Face Animation')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=1000, help='number of epochs')
    parser.add_argument("--emo_dim", type=int, default=7, help='dimension of emotions')
    parser.add_argument("--out_dim", type=int, default=174, help='dimension of output controller value')
    parser.add_argument("--save_path", type=str, default="weights", help='path of the trained models')
    parser.add_argument("--train_loader", type=str, default="train_loader.pth", help='path of train dataloader')
    parser.add_argument("--valid_loader", type=str, default="valid_loader.pth", help='path of valid dataloader')
    args = parser.parse_args()

    # build model
    model = EmoFace(emo_dim=args.emo_dim, out_dim=args.out_dim).to(device)
    print("model parameters: ", count_parameters(model))

    # to cuda
    lr = 1e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.995)
    train_loader = torch.load('train_loader.pth')
    valid_loader = torch.load('valid_loader.pth')

    writer = tensorboardX.SummaryWriter(comment='EmoFace')

    trainer(args, train_loader, valid_loader, model, optimizer, scheduler, args.max_epoch, writer)


if __name__ == "__main__":
    main()
