#!/usr/bin/env bash
set -e

output_root="${LEGS_OUTPUT_ROOT:-output/rl}"
if [[ -e "$output_root" || -L "$output_root" ]]; then
    printf 'Output directory already exists: %s. Set LEGS_OUTPUT_ROOT to a fresh path.\n' "$output_root" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$output_root")"
mkdir -- "$output_root"

echo "mipnerf开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=bicycle python train.py -s /data2/ningzhh/data/mipnerf360/bicycle -m "$output_root/bicycle" -i images_4 --eval --densification_interval 100  --optimizer_type default  --grad_abs_thresh 0.0004
OAR_JOB_ID=flowers python train.py -s /data2/ningzhh/data/mipnerf360/flowers -m "$output_root/flowers" -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004
OAR_JOB_ID=garden python train.py -s /data2/ningzhh/data/mipnerf360/garden -m "$output_root/garden" -i images_4 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02  --grad_abs_thresh 0.0002
OAR_JOB_ID=stump python train.py -s /data2/ningzhh/data/mipnerf360/stump -m "$output_root/stump" -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004
OAR_JOB_ID=treehill python train.py -s /data2/ningzhh/data/mipnerf360/treehill -m "$output_root/treehill" -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0005
OAR_JOB_ID=room python train.py -s /data2/ningzhh/data/mipnerf360/room -m "$output_root/room" -i images_2 --eval --densification_interval 100  --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002
OAR_JOB_ID=counter python train.py -s /data2/ningzhh/data/mipnerf360/counter -m "$output_root/counter" -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0002
OAR_JOB_ID=kitchen python train.py -s /data2/ningzhh/data/mipnerf360/kitchen -m "$output_root/kitchen" -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001
OAR_JOB_ID=bonsai python train.py -s /data2/ningzhh/data/mipnerf360/bonsai -m "$output_root/bonsai" -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001
end=$(date +%s)
echo "mipnerf结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "mipnerf耗时: $((end - start)) s"

echo "tanksandtemples开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=truck python train.py -s /data2/ningzhh/data/tanksandtemples/truck -m "$output_root/truck" --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.04 --grad_abs_thresh 0.0001 --mult 0.7
OAR_JOB_ID=train python train.py -s /data2/ningzhh/data/tanksandtemples/train -m "$output_root/train" --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.042 --grad_abs_thresh 0.0001 --mult 0.7
end=$(date +%s)
echo "tanksandtemples结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "tanksandtemples耗时: $((end - start)) s"

echo "db开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=drjohnson python train.py -s /data2/ningzhh/data/db/drjohnson -m "$output_root/drjohnson" --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0002 --mult 0.7
OAR_JOB_ID=playroom python train.py -s /data2/ningzhh/data/db/playroom -m "$output_root/playroom" --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0015 --grad_abs_thresh 0.0002 --mult 0.7
end=$(date +%s)
echo "db结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "db耗时: $((end - start)) s"


python render.py -m "$output_root/bicycle" --skip_train
python metrics.py -m "$output_root/bicycle"

python render.py -m "$output_root/flowers" --skip_train
python metrics.py -m "$output_root/flowers"

python render.py -m "$output_root/garden" --skip_train
python metrics.py -m "$output_root/garden"

python render.py -m "$output_root/stump" --skip_train
python metrics.py -m "$output_root/stump"

python render.py -m "$output_root/treehill" --skip_train
python metrics.py -m "$output_root/treehill"

python render.py -m "$output_root/room" --skip_train
python metrics.py -m "$output_root/room"

python render.py -m "$output_root/counter" --skip_train
python metrics.py -m "$output_root/counter"

python render.py -m "$output_root/kitchen" --skip_train
python metrics.py -m "$output_root/kitchen"

python render.py -m "$output_root/bonsai" --skip_train
python metrics.py -m "$output_root/bonsai"

python render.py -m "$output_root/truck" --skip_train --mult 0.7
python metrics.py -m "$output_root/truck"

python render.py -m "$output_root/train" --skip_train --mult 0.7
python metrics.py -m "$output_root/train"

python render.py -m "$output_root/drjohnson" --skip_train --mult 0.7
python metrics.py -m "$output_root/drjohnson"

python render.py -m "$output_root/playroom" --skip_train --mult 0.7
python metrics.py -m "$output_root/playroom"

python report_results.py
