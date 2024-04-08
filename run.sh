#!/usr/bin/env bash
set -e
scenes=(hellwarrior  hook  jumpingjacks lego  mutant  standup  trex)
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
    scene=${scenes[i]}
    gpu_id=${gpus[$(( i % num_gpus ))]}
    echo "use gpu${gpu_id} on scene: ${scene} "
    screen -S gpu${gpu_id} -p 0 -X stuff "^M"
    if [[ ! -e ./logs/dnerf/${scene}/temporalpoints_last.tar ]]
    then
      screen -S gpu${gpu_id} -p 0 -X stuff \
        "python run.py --config configs/nerf/${scene}.py --i_print 1000 --render_video --render_pcd ^M"
    fi
    screen -S gpu${gpu_id} -p 0 -X stuff "python test.py --config configs/nerf/${scene}.py --num_fps=200 ^M"
done

#    screen -S gpu${gpu_id} -p 0 -X stuff \
#      "python run.py --config configs/nerf/${scene}.py --i_print 1000 --render_video --render_only ^M"
#    screen -S gpu${gpu_id} -p 0 -X stuff \
#      "python run.py --config configs/nerf/${scene}.py --i_print 1000 --render_video --render_only --render_pcd ^M"
#    screen -S gpu${gpu_id} -p 0 -X stuff \
#      "python run.py --config configs/nerf/${scene}.py --i_print 1000 --visualise_canonical --render_pcd --render_only ^M"
#    screen -S gpu${gpu_id} -p 0 -X stuff \
#      "python run.py --config configs/nerf/${scene}.py --i_print 1000 --visualise_canonical --render_pcd --render_only --degree_threshold 30 ^M"
#    screen -S gpu${gpu_id} -p 0 -X stuff \
#      "python run.py --config configs/nerf/${scene}.py --i_print 1000 --repose_pcd --render_only --render_pcd --degree_threshold 30 ^M"