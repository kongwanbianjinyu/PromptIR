import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F

from net.camixer_prompt_xrestormer_effv2 import CAPromptXRestormerEffv2



class CAPromptXRestormerEffv2IRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.net =  CAPromptXRestormerEffv2(
                    inp_channels=3,
                    out_channels=3,
                    dim = 48,
                    num_blocks = [2,4,4,4],
                    num_refinement_blocks = 4,
                    channel_heads = [1,1,1,1],
                    spatial_heads = [1,2,4,8],
                    overlap_ratio = 0.5,
                    dim_head = 16,
                    ratio = 0.5,
                    window_size = 8,
                    bias = False,
                    ffn_expansion_factor = 2.66,
                    LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
                    dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                    scale = 1,
                    prompt = True
                    )
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x, training = True):
        return self.net(x, training = training)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, ratio_loss = self.net(degrad_patch, training = True)

        loss = self.loss_fn(restored,clean_patch)  + ratio_loss
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
    denoise_splits = "bsd68/"
    derain_splits = "Rain100L/"

    denoise_testset = DenoiseTestDataset(opt)

    print(f'Start {denoise_splits} testing Sigma=15...')
    psnr_denoise_15, ssim_denoise_15  = test_Denoise(opt, net, denoise_testset, sigma=15)
    logger.log_metrics({"PSNR_Denoise_15":psnr_denoise_15,"SSIM_Denoise_15":ssim_denoise_15})

    print(f'Start {denoise_splits} testing Sigma=25...')
    psnr_denoise_25, ssim_denoise_25 = test_Denoise(opt, net, denoise_testset, sigma=25)
    logger.log_metrics({"PSNR_Denoise_25":psnr_denoise_25,"SSIM_Denoise_25":ssim_denoise_25})

    print(f'Start {denoise_splits} testing Sigma=50...')
    psnr_denoise_50, ssim_denoise_50 = test_Denoise(opt, net, denoise_testset, sigma=50)
    logger.log_metrics({"PSNR_Denoise_50":psnr_denoise_50,"SSIM_Denoise_50":ssim_denoise_50})


    print(f'Start testing {derain_splits} rain streak removal...')
    derain_set = DerainDehazeDataset(opt, addnoise=False, sigma=15)
    psnr_derain, ssim_derain = test_Derain_Dehaze(opt, net, derain_set, task="derain")
    logger.log_metrics({"PSNR_Derain":psnr_derain,"SSIM_Derain":ssim_derain})

    print('Start testing SOTS...')
    psnr_dehaze, ssim_dehaze = test_Derain_Dehaze(opt, net, derain_set, task="dehaze")
    logger.log_metrics({"PSNR_Dehaze":psnr_dehaze,"SSIM_Dehaze":ssim_dehaze})


def test_Denoise(testopt, net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            # degrad_patch must be mutipler of 64
            _, _, H_old, W_old = degrad_patch.shape
            h_pad = (H_old // 64 + 1) * 64 - H_old
            w_pad = (W_old // 64 + 1) * 64 - W_old
            degrad_patch = torch.cat([degrad_patch, torch.flip(degrad_patch, [2])], 2)[:,:,:H_old+h_pad,:]
            degrad_patch = torch.cat([degrad_patch, torch.flip(degrad_patch, [3])], 3)[:,:,:,:W_old+w_pad]

            #restored = net(degrad_patch)
            restored = net(degrad_patch, training = False)
            restored = restored[:,:,:H_old,:W_old]
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
    return psnr.avg, ssim.avg



def test_Derain_Dehaze(testopt, net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

             # degrad_patch must be mutipler of 64
            _, _, H_old, W_old = degrad_patch.shape
            
            h_pad = (H_old // 64 + 1) * 64 - H_old
            w_pad = (W_old // 64 + 1) * 64 - W_old
            degrad_patch = torch.cat([degrad_patch, torch.flip(degrad_patch, [2])], 2)[:,:,:H_old+h_pad,:]
            degrad_patch = torch.cat([degrad_patch, torch.flip(degrad_patch, [3])], 3)[:,:,:,:W_old+w_pad]

            #restored = net(degrad_patch)
            restored = net(degrad_patch,training = False)
            restored = restored[:,:,:H_old:,:W_old]

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

    return psnr.avg, ssim.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="/data/jiachen/all_in_one/Test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="/data/jiachen/all_in_one/Test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="/data/jiachen/all_in_one/Test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output_ca_eff_infer/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="/home/jiachen/PromptIR/ckpt/epoch=51-step=617188.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path = testopt.ckpt_name


    
    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)


    print("CKPT name : {}".format(ckpt_path))

    net  = CAPromptXRestormerEffv2IRModel().cuda()#.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    
    if testopt.mode == 0:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(testopt, net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(testopt, net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(testopt, net, testset, sigma=50)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(testopt, net, derain_set, task="derain")
    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        test_Derain_Dehaze(testopt, net, derain_set, task="dehaze")
    elif testopt.mode == 3:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            psnr_denoise_15, ssim_denoise_15 = test_Denoise(testopt, net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            psnr_denoise_25, ssim_denoise_25 = test_Denoise(testopt, net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            psnr_denoise_50, ssim_denoise_50 = test_Denoise(testopt, net, testset, sigma=50)



        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            psnr_derain, ssim_derain = test_Derain_Dehaze(testopt, net, derain_set, task="derain")

        print('Start testing SOTS...')
        psnr_dehaze, ssim_dehaze = test_Derain_Dehaze(testopt, net, derain_set, task="dehaze")