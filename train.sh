#python train.py --model xrestormerir --output_path xrestormerir/output/ --ckpt_dir xrestormerir/train_ckpt/ --batch_size 4

# train promptxrestormereffir
#python train.py --model promptxrestormereffir --output_path promptxrestormereffir/output/ --ckpt_dir promptxrestormereffir/train_ckpt/ --batch_size 4

# train promptxrestormereffv2ir
python train_capromptxrestormer.py --model CAPromptXRestormerEffIRv2 --output_path capromptxrestormerv2_new/output/ --ckpt_dir capromptxrestormerv2_new/train_ckpt/ --batch_size 4 --wblogger camixerpromptir