export CUDA_VISIBLE_DEVICES=1,2,3,0
PYTHONPATH=`pwd`/../fairseq \
--config-dir conf/seed1337 \
--config-name multi_target_avhubert \
hydra.run.dir=`pwd`/exp/ \
common.user_dir=`pwd` \
model.w2v_path=`pwd`/checkpoints/large_vox_iter5.pt \
task.label_dir=data/label  \
task.data=data/label  \













