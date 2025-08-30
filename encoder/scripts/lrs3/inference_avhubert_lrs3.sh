CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=`pwd`/../fairseq \
python -B inference.py \
--config-dir conf \
--config-name decode \
hydra.run.dir=`pwd` \
common.user_dir=`pwd` \
common_eval.path=`pwd`/exp/0215_100spk_3sprompt_L1_yourTTSspk/checkpoint_last.pt \
common_eval.results_path=`pwd`/../results/lrs2/test \
override.w2v_path=`pwd`/checkpoints/large_vox_iter5.pt \
override.label_dir=/LRS3-TED/dataset4ISU/data/label_eval_in_train  \
override.data=/LRS3-TED/dataset4ISU/data/label_eval_in_train  \

