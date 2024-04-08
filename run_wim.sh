#!/usr/bin/env bash
set -e

scenes=(atlas baxter cassie iiwa nao pandas spot)
gpus=(4)
args=()
test_args=()
num_scenes=${#scenes[@]}
num_gpus=${#gpus[@]}
echo "There are ${num_gpus} gpus and ${num_scenes} scenes"

for (( i = 0;  i < ${num_gpus}; ++i ))
do
    gpu_id="gpu${gpus[$i]}"
    if ! screen -ls ${gpu_id}
    then
        echo "create ${gpu_id}"
        screen -dmS ${gpu_id}
    fi
    screen -S ${gpu_id} -p 0 -X stuff "^M"
    screen -S ${gpu_id} -p 0 -X stuff "export CUDA_VISIBLE_DEVICES=${gpus[$i]}^M"
    screen -S ${gpu_id} -p 0 -X stuff "cd ~/Projects/NeRF/Articulated-Point-NeRF^M"
done
screen -ls%

for (( i=0; i < num_scenes; ++i ))
do
    gpu_id=${gpus[$(( i % num_gpus ))]}
    scene=${scenes[i]}
    echo "use gpu${gpu_id} on scene: ${scene} "
    screen -S gpu${gpu_id} -p 0 -X stuff "^M"
    if [ ! -e logs/wim/${scene}/temporalpoints_last.tar ]
    then
      screen -S gpu${gpu_id} -p 0 -X stuff \
        "python run.py --config configs/wim/${scene}.py --i_print 1000 --render_video --render_pcd ^M"
    fi
    screen -S gpu${gpu_id} -p 0 -X stuff "python test.py --config configs/wim/${scene}.py --num_fps=-1 ^M"
done
