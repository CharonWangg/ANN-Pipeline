#! /bin/bash
PYTHON=/home/charon/anaconda3/envs/ml37/bin/python
JOBS=4
DELAY=5
# Next: width: 10; seed: 7, 42, 1234, 2022, 3057
# data
DATASET=(cifar10)
# model
MODEL_NAME=(resnet_he)
DEPTH=(14)
WIDTH_MULTIPLIER=(1)
# optimization
LR=(0.12)
MAX_EPOCHS=(1)
OPTIMIZER=(SGD)
MOMENTUM=(0.9)
LR_SCHEDULER=(cifar)
TRAIN_BATCH_SIZE=(32 64 128 256)
VALID_BATCH_SIZE=(256)
WEIGHT_DECAY=(0.0005)
# augmentation
AUG=(True)
AUG_PROB=(0.5)
# hardware
STRATEGY=None
PRECISION=(32)
GPUS=(3)
# randomness
SEED=(7)
# logging
VAL_CHECK_INTERVAL=(10)
EXP_NAME=(resnet_he_cifar10_intermediate)
RUN=(0)


parallel --delay=$DELAY --linebuffer --jobs=$JOBS $PYTHON train_by_cmd.py --dataset={1} --model_name={2} --depth={3} --width_multiplier={4} \
--lr={5} --max_epochs={6} --optimizer={7} --momentum={8} --lr_scheduler={9} --train_batch_size={10} --weight_decay={11} \
--aug={12} --aug_prob={13} --precision={14} --seed={15} --strategy={16} --gpus={17} --exp_name={18} --run={19} --valid_batch_size={20} \
--val_check_interval={21} \
  ::: ${DATASET[@]} ::: ${MODEL_NAME[@]} ::: ${DEPTH[@]} ::: ${WIDTH_MULTIPLIER[@]} ::: ${LR[@]} ::: ${MAX_EPOCHS[@]} \
  ::: ${OPTIMIZER[@]} ::: ${MOMENTUM[@]} ::: ${LR_SCHEDULER[@]} ::: ${TRAIN_BATCH_SIZE[@]} ::: ${WEIGHT_DECAY[@]} \
  ::: ${AUG[@]} ::: ${AUG_PROB[@]} ::: ${PRECISION[@]} ::: ${SEED[@]} ::: ${STRATEGY} ::: ${GPUS[@]} ::: ${EXP_NAME[@]} ::: ${RUN[@]} \
  ::: ${VALID_BATCH_SIZE[@]} ::: ${VAL_CHECK_INTERVAL[@]}

