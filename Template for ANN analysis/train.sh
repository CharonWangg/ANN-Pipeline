#! /bin/bash
PYTHON=/home/charon/anaconda3/envs/ml37/bin/python
JOBS=1

# data
DATASET=(cifar10)
# model
MODEL_NAME=(resnet_he)
DEPTH=(20)
WIDTH_MULTIPLIER=(1)
# optimization
LR=(0.1)
MAX_EPOCHS=(30)
OPTIMIZER=(SGD)
MOMENTUM=(0.9)
LR_SCHEDULER=(cifar)
TRAIN_BATCH_SIZE=(256)
WEIGHT_DECAY=(0.0005)
# augmentation
AUG=(True)
AUG_PROB=(0.5)
# hardware
STRATEGY=None
PRECISION=(32)
DEVICES=([0])
# randomness
SEED=(7)
# logging
EXP_NAME=(resnet_he_cifar10)
RUN=(0)


parallel --linebuffer --jobs=$JOBS $PYTHON train_by_cmd.py --dataset={1} --model_name={2} --depth={3} --width_multiplier={4} \
--lr={5} --max_epochs={6} --optimizer={7} --momentum={8} --lr_scheduler={9} --train_batch_size={10} --weight_decay={11} \
--aug={12} --aug_prob={13} --precision={14} --seed={15} --strategy={16} --devices={17} --exp_name={18} --run={19} \
  ::: ${DATASET[@]} ::: ${MODEL_NAME[@]} ::: ${DEPTH[@]} ::: ${WIDTH_MULTIPLIER[@]} ::: ${LR[@]} ::: ${MAX_EPOCHS[@]} \
  ::: ${OPTIMIZER[@]} ::: ${MOMENTUM[@]} ::: ${LR_SCHEDULER[@]} ::: ${TRAIN_BATCH_SIZE[@]} ::: ${WEIGHT_DECAY[@]} \
  ::: ${AUG[@]} ::: ${AUG_PROB[@]} ::: ${PRECISION[@]} ::: ${SEED[@]} ::: ${STRATEGY} ::: ${DEVICES[@]} ::: ${EXP_NAME[@]} ::: ${RUN[@]}

