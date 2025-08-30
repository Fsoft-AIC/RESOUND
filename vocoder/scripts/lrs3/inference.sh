export CUDA_VISIBLE_DEVICES=2
python inference.py \
--code_file ../results/lrs2/test/pred_unit  \
--output_dir ../results/lrs2/test  \
--checkpoint_file `pwd`/checkpoints/vocoder_lrs3_aug_multi.pt \
-n -1 

