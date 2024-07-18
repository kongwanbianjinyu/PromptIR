import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--model', type=str, default='promptir', help='which model to use.')
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=6,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='/data/jiachen/all_in_one/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='/data/jiachen/all_in_one/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='/data/jiachen/all_in_one/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/all_in_one/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 4,help = "Number of GPUs to use for training")


# Evaluate Path
parser.add_argument('--denoise_path', type=str, default="/data/jiachen/all_in_one/Test/denoise/bsd68/", help='save path of test noisy images')
parser.add_argument('--derain_path', type=str, default="/data/jiachen/all_in_one/Test/derain/Rain100L/", help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default="/data/jiachen/all_in_one/Test/dehaze/", help='save path of test hazy images')

options = parser.parse_args()

