from utils.val_utils import AverageMeter, compute_psnr_ssim
from skimage.io import imread
import os
import torch


denoise15_restored_path = "/data/jiachen/all_in_one/visual_results/denoise/15"
denoise_clean_path = "/data/jiachen/all_in_one/Test/denoise/bsd68"

restored_images = sorted(os.listdir(denoise15_restored_path))
clean_images = sorted(os.listdir(denoise_clean_path))

print(restored_images)
print(clean_images)


# for each image in the denoise15_restored_path and denoise_clean_path
# compute the PSNR and SSIM

for restored_image, clean_image in zip(restored_images, clean_images):
    psnr = AverageMeter()
    ssim = AverageMeter()

    restored_image_path = os.path.join(denoise15_restored_path, restored_image)
    clean_image_path = os.path.join(denoise_clean_path, clean_image)
    
    # Read images
    restored = imread(restored_image_path) # (320, 480, 3)
    clean = imread(clean_image_path) # (321, 481, 3)

    # crop clean to the same size as clean
    clean = clean[:320, :480, :]

    restored = torch.from_numpy(restored).permute(2, 0, 1).float() / 255.0  # Convert to [C, H, W] and normalize
    clean = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0  # Convert to [C, H, W] and normalize

    restored.unsqueeze_(0)
    clean.unsqueeze_(0)

    print(restored.shape)
    print(clean.shape)

    temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean)
    psnr.update(temp_psnr, N)
    ssim.update(temp_ssim, N)

print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))