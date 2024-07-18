import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from net.prompt_uformer import PromptUformerIR
from net.prompt_xrestormer import PromptXRestormer
from net.prompt_xrestormer_eff import PromptXRestormerEff
from net.xrestormer import XRestormer
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import os
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from test import *


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]


class PromptUformerIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptUformerIR(embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True)  
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]  
    

class PromptXRestormerIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptXRestormer(
                    inp_channels=3,
                    out_channels=3,
                    dim = 48,
                    num_blocks = [2,4,4,4],
                    num_refinement_blocks = 4,
                    channel_heads= [1,1,1,1],
                    spatial_heads= [1,2,4,8],
                    overlap_ratio= [0.5, 0.5, 0.5, 0.5],
                    ffn_expansion_factor = 2.66,
                    bias = False,
                    LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
                    dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                    scale = 1,prompt = True
                    )
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]  
def evaluate_model(net, opt, logger):
    net = net  # Assuming net is your model
    denoise_splits = "bsd68/"
    derain_splits = "Rain100L/"

    denoise_testset = DenoiseTestDataset(opt)

    print(f'Start {denoise_splits} testing Sigma=15...')
    psnr_denoise_15, ssim_denoise_15  = test_Denoise(opt, net, denoise_testset, sigma=15)
    logger.log_metrics({"PSNR_Denoise_15":psnr_denoise_15,"SSIM_Denoise_15":ssim_denoise_15})

    # print(f'Start {denoise_splits} testing Sigma=25...')
    # psnr_denoise_25, ssim_denoise_25 = test_Denoise(opt, net, denoise_testset, sigma=25)
    # logger.log_metrics({"PSNR_Denoise_25":psnr_denoise_25,"SSIM_Denoise_25":ssim_denoise_25})

    # print(f'Start {denoise_splits} testing Sigma=50...')
    # psnr_denoise_50, ssim_denoise_50 = test_Denoise(opt, net, denoise_testset, sigma=50)
    # logger.log_metrics({"PSNR_Denoise_50":psnr_denoise_50,"SSIM_Denoise_50":ssim_denoise_50})


    print(f'Start testing {derain_splits} rain streak removal...')
    derain_set = DerainDehazeDataset(opt, addnoise=False, sigma=15)
    psnr_derain, ssim_derain = test_Derain_Dehaze(opt, net, derain_set, task="derain")
    logger.log_metrics({"PSNR_Derain":psnr_derain,"SSIM_Derain":ssim_derain})

    # print('Start testing SOTS...')
    # psnr_dehaze, ssim_dehaze = test_Derain_Dehaze(opt, net, derain_set, task="dehaze")
    # logger.log_metrics({"PSNR_Dehaze":psnr_dehaze,"SSIM_Dehaze":ssim_dehaze})



class EvaluationCallback(pl.Callback):
    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        print("Evaluating Model...")
        evaluate_model(pl_module, self.opt, self.logger)

class PromptXRestormerEffIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptXRestormerEff(
                    inp_channels=3,
                    out_channels=3,
                    dim = 48,
                    num_blocks = [2,4,4,4],
                    num_refinement_blocks = 4,
                    channel_heads= [1,1,1,1],
                    spatial_heads= [1,2,4,8],
                    overlap_ratio= [0.5, 0.5, 0.5, 0.5],
                    ffn_expansion_factor = 2.66,
                    bias = False,
                    LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
                    dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                    scale = 1,prompt = True
                    )
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]  
    

class XRestormerIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = XRestormer(
                    inp_channels=3,
                    out_channels=3,
                    dim = 48,
                    num_blocks = [2,4,4,4],
                    num_refinement_blocks = 4,
                    channel_heads= [1,1,1,1],
                    spatial_heads= [1,2,4,8],
                    overlap_ratio= [0.5, 0.5, 0.5, 0.5],
                    ffn_expansion_factor = 2.66,
                    bias = False,
                    LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
                    dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                    scale = 1
                    )
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]  
    
def evaluate_model(net, opt, logger):
    net = net  # Assuming net is your model
    denoise_splits = "bsd68/"
    derain_splits = "Rain100L/"

    denoise_testset = DenoiseTestDataset(opt)

    print(f'Start {denoise_splits} testing Sigma=15...')
    psnr_denoise_15, ssim_denoise_15  = test_Denoise(opt, net, denoise_testset, sigma=15)
    logger.log_metrics({"PSNR_Denoise_15":psnr_denoise_15,"SSIM_Denoise_15":ssim_denoise_15})

    # print(f'Start {denoise_splits} testing Sigma=25...')
    # psnr_denoise_25, ssim_denoise_25 = test_Denoise(opt, net, denoise_testset, sigma=25)
    # logger.log_metrics({"PSNR_Denoise_25":psnr_denoise_25,"SSIM_Denoise_25":ssim_denoise_25})

    # print(f'Start {denoise_splits} testing Sigma=50...')
    # psnr_denoise_50, ssim_denoise_50 = test_Denoise(opt, net, denoise_testset, sigma=50)
    # logger.log_metrics({"PSNR_Denoise_50":psnr_denoise_50,"SSIM_Denoise_50":ssim_denoise_50})


    print(f'Start testing {derain_splits} rain streak removal...')
    derain_set = DerainDehazeDataset(opt, addnoise=False, sigma=15)
    psnr_derain, ssim_derain = test_Derain_Dehaze(opt, net, derain_set, task="derain")
    logger.log_metrics({"PSNR_Derain":psnr_derain,"SSIM_Derain":ssim_derain})

    # print('Start testing SOTS...')
    # psnr_dehaze, ssim_dehaze = test_Derain_Dehaze(opt, net, derain_set, task="dehaze")
    # logger.log_metrics({"PSNR_Dehaze":psnr_dehaze,"SSIM_Dehaze":ssim_dehaze})



class EvaluationCallback(pl.Callback):
    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        print("Evaluating Model...")
        evaluate_model(pl_module, self.opt, self.logger)





def main():
    print("Options")
    print(opt)
    if opt.model == "promptir":
        exp_name = "PromptIR"
        model = PromptIRModel()
    elif opt.model == "promptuformerir":
        exp_name = "PromptUformerIR"
        model = PromptUformerIRModel()
    elif opt.model == "promptxrestormerir":
        exp_name = "PromptXRestormerIR"
        model = PromptXRestormerIRModel()

    elif opt.model == "promptxrestormereffir":
        exp_name = "PromptXRestormerEffIR"
        model = PromptXRestormerEffIRModel()

    elif opt.model == "xrestormerir":
        exp_name = "XRestormerIR"
        model = XRestormerIRModel()

    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name=exp_name)
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    evaluate_callback = EvaluationCallback(opt, logger)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    trainer = pl.Trainer(max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",
                         logger=logger,callbacks=[checkpoint_callback, evaluate_callback]) # [DEBUG]: ,limit_train_batches=0.001
    trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path = "/home/jiachen/PromptIR/promptxrestormereffir/train_ckpt/epoch=62-step=560826.ckpt")


if __name__ == '__main__':
    main()



