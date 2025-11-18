#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 2
#SBATCH --time=0-120
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --array=0-0
##SBATCH --killable
##SBATCH --requeue
#SBATCH --output /cs/labs/daphna/itai.david/py_repos/TypiClust/run_outputs/run_exp_%A_%a.txt
## gpu:rtx2080:1, gpu:a10:1
dir=/cs/labs/daphna/itai.david/py_repos/TypiClust/deep-al/tools



cd $dir

source /cs/labs/daphna/itai.david/envs/venv_2705/bin/activate



##
#COMMAND_TO_RUN=$(python /cs/labs/daphna/itai.david/py_repos/python_runners/bayes_misp_runner.py ${SLURM_ARRAY_TASK_ID})
#eval "$COMMAND_TO_RUN"

##python  ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al all_misp --exp-name auto --initial_size 0 --budget 100 --initial_delta 1.5 --seed $SLURM_ARRAY_TASK_ID --eval_model_type from_features --kernel_type rbf --diff_method prob_method_v1
##python  ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al all_misp --exp-name auto --initial_size 0 --budget 100 --initial_delta 1.75 --seed $SLURM_ARRAY_TASK_ID --eval_model_type from_features --kernel_type rbf --diff_method prob_method_v1
##python  ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al all_misp --exp-name auto --initial_size 0 --budget 100 --initial_delta 0.5 --seed $SLURM_ARRAY_TASK_ID --eval_model_type from_features --kernel_type rbf --diff_method prob_method_v1
##python  ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al all_misp --exp-name auto --initial_size 0 --budget 100 --initial_delta 1 --seed $SLURM_ARRAY_TASK_ID --eval_model_type from_features --kernel_type rbf --diff_method combine_uncert_type_outer_mean
##python  ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al all_misp --exp-name auto --initial_size 0 --budget 100 --initial_delta 0.65 --seed $SLURM_ARRAY_TASK_ID --eval_model_type from_features --kernel_type tophat --diff_method 2_closest_diff
##python  ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al all_misp --exp-name auto --initial_size 0 --budget 100 --initial_delta 0.65 --seed $SLURM_ARRAY_TASK_ID --eval_model_type from_features --kernel_type tophat --diff_method 2_closest_diff --max_iter 100 --high_budget --confidence_method max
##python  ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al uncertainty --exp-name auto --initial_size 0 --budget 100 --initial_delta 0.01 --seed $SLURM_ARRAY_TASK_ID --eval_model_type from_features --kernel_type rbf --diff_method 2_closest_diff --max_iter 100 --high_budget
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al dcom --exp-name dcom_from_features_full --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al prob_cover --exp-name prob_cover_from_features_full --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/half_moon/al/RESNET18.yaml --al random --exp-name random_from_features --initial_size 0 --budget 2 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/scenario_a/al/RESNET18.yaml --al random --exp-name random_from_features --initial_size 0 --budget 5 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/half_moon/al/RESNET18.yaml --al prob_cover --exp-name prob_cover_from_features --initial_size 0 --budget 2 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/scenario_a/al/RESNET18.yaml --al prob_cover --exp-name prob_cover_from_features --initial_size 0 --budget 5 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/half_moon/al/RESNET18.yaml --al dcom --exp-name dcom_from_features --initial_size 0 --budget 2 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/scenario_a/al/RESNET18.yaml --al dcom --exp-name dcom_from_features --initial_size 0 --budget 5 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/half_moon/al/RESNET18.yaml --al margin --exp-name margin_from_features --initial_size 0 --budget 2 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/scenario_a/al/RESNET18.yaml --al margin --exp-name margin_from_features --initial_size 0 --budget 5 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al margin --exp-name margin_from_features --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID --sparse_ds

## CIFAR 100 REF RUNS SPARSE
#python ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al margin --exp-name margin_from_features --initial_size 0 --budget 100 --initial_delta 0.01 --seed $SLURM_ARRAY_TASK_ID --eval_model from_features --sparse_ds
#python ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al random --exp-name random_from_features --initial_size 0 --budget 100 --initial_delta 0.01 --seed $SLURM_ARRAY_TASK_ID --eval_model from_features --sparse_ds
#python ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al dcom --exp-name dcom_from_features --initial_size 0 --budget 100 --initial_delta 0.65 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID --sparse_ds
#python ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al prob_cover --exp-name prob_cover_from_features --initial_size 0 --budget 100 --initial_delta 0.65 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID --sparse_ds

## CIFAR 100 REF RUNS
#python ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al margin --exp-name margin_from_features --initial_size 0 --budget 100 --initial_delta 0.01 --seed $SLURM_ARRAY_TASK_ID --eval_model from_features
#python ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al random --exp-name random_from_features --initial_size 0 --budget 100 --initial_delta 0.01 --seed $SLURM_ARRAY_TASK_ID --eval_model from_features
#python ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al dcom --exp-name dcom_from_features --initial_size 0 --budget 100 --initial_delta 0.65 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/cifar100/al/RESNET18.yaml --al prob_cover --exp-name prob_cover_from_features --initial_size 0 --budget 100 --initial_delta 0.55 --eval_model from_images --seed $SLURM_ARRAY_TASK_ID




## CIFAR 10 REF RUNS
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al dcom --exp-name dcom_from_images --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al prob_cover --exp-name prob_cover_from_images --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --seed $SLURM_ARRAY_TASK_ID
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al margin --exp-name margin_from_images --initial_size 0 --budget 10 --initial_delta 0.01 --seed $SLURM_ARRAY_TASK_ID --eval_model from_images
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al random --exp-name random_from_images --initial_size 0 --budget 10 --initial_delta 0.01 --seed $SLURM_ARRAY_TASK_ID --eval_model from_images
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al max_herding --exp-name max_herding_from_images --initial_size 0 --budget 10 --initial_delta 1.0 --seed $SLURM_ARRAY_TASK_ID --eval_model from_images
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name max_herding_from_images --initial_size 0 --budget 10 --initial_delta 1.0 --seed $SLURM_ARRAY_TASK_ID --eval_model from_images --diff_method max_herding --kernel_type rbf --alpha 0.0
#

## CIFAR 10 REF RUNS SPARSE
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al margin --exp-name margin_from_features --initial_size 0 --budget 10 --initial_delta 0.01 --seed $SLURM_ARRAY_TASK_ID --eval_model from_features --sparse_ds
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al random --exp-name random_from_features --initial_size 0 --budget 10 --initial_delta 0.01 --seed $SLURM_ARRAY_TASK_ID --eval_model from_features --sparse_ds
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al dcom --exp-name dcom_from_features --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID --sparse_ds
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al prob_cover --exp-name prob_cover_from_features --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_features --seed $SLURM_ARRAY_TASK_ID --sparse_ds


### HIGH BUDGET EXPS
python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name fixed_bayes_misp_full_test_fusion_cont_2910 --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --kernel_type tophat --diff_method full_weighted_max --alpha 0.1 --cont_method abs --seed 0 --max_iter 300
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name fixed_bayes_misp_full_test_fusion_cont_2910 --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --kernel_type tophat --diff_method max --alpha 0.1 --cont_method positive --seed 0 --max_iter 300
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name fixed_bayes_misp_full_test_fusion_cont_2910 --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --kernel_type tophat --diff_method full_weighted_max --alpha 1 --cont_method positive --seed 0 --max_iter 500
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name fixed_bayes_misp_full_test_fusion_cont_2910 --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --kernel_type tophat --diff_method max --alpha 0.0001 --cont_method positive --seed 0 --max_iter 500
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name fixed_bayes_misp_full_test_fusion_cont_2910 --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --kernel_type tophat --diff_method full_weighted_max --alpha 0.001 --cont_method positive --seed 0 --max_iter 750
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name fixed_bayes_misp_full_test_fusion_cont_2910 --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --kernel_type tophat --diff_method max --alpha 0.001 --cont_method positive --seed 0 --max_iter 750
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al prob_cover --exp-name prob_cover_from_images --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --seed 0 --max_iter 300
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al dcom --exp-name dcom_from_images --initial_size 0 --budget 10 --initial_delta 0.75 --eval_model from_images --seed 2 --max_iter 300
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al random --exp-name random_from_images --initial_size 0 --budget 10 --initial_delta 0.01 --seed 0 --eval_model from_images --max_iter 750
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al margin --exp-name margin_from_images --initial_size 0 --budget 10 --initial_delta 0.01 --seed 0 --eval_model from_images --max_iter 300
#python ./train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al bayes_misp --exp-name max_herding_from_images --initial_size 0 --budget 10 --initial_delta 1.0 --seed 2 --eval_model from_images --diff_method max_herding --kernel_type rbf --alpha 0.0 --max_iter 300
