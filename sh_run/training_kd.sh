export dataset=cifar100
NUM_GPUS=2
GPU_ID=2,3
method=kd


for temp in 4
do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
    --master_addr 127.0.0.2 --master_port 29501 \
    --use_env examples/image_classification.py \
    --config configs/sample/${dataset}/${method}/wide_resnet40_1_from_wide_resnet40_4-final_run.yaml \
    --log log/${dataset}/${method}/wide_resnet40_1_from_wide_resnet40_4/wide_resnet40_1_from_wide_resnet40_4-final_runlog \
    --temp ${temp} --alpha 0.5 \
    -test_only
    # -adjust_lr --world_size ${NUM_GPUS} 
done

