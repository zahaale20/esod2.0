model=${MODEL:-"yolov5m"}
dataset=${DATASET:-"visdrone"}
gpus=${GPUS:-"0"}
bsz=${BATCH_SIZE:-8}
imsz=${IMAGE_SIZE:-1536}
epochs=${EPOCHS:-50}

if [[ ${gpus} == *,* ]]; then
    ngpus=$(($(echo "$gpus" | grep -o ',' | wc -l) + 1))
    python -m torch.distributed.launch --nproc_per_node ${ngpus} \
           train.py --data data/${dataset}.yaml \
                    --cfg models/cfg/esod_${model}_${dataset}.yaml \
                    --weights weights/${model}.pt \
                    --hyp data/hyp.${dataset}.yaml \
                    --batch-size ${bsz} \
                    --img-size ${imsz} \
                    --epochs ${epochs} \
                    --device ${gpus}
else
    python train.py --data data/${dataset}.yaml \
                    --cfg models/cfg/esod_${model}_${dataset}.yaml \
                    --weights weights/${model}.pt \
                    --hyp data/hyp.${dataset}.yaml \
                    --batch-size ${bsz} \
                    --img-size ${imsz} \
                    --epochs ${epochs} \
                    --device ${gpus}
fi