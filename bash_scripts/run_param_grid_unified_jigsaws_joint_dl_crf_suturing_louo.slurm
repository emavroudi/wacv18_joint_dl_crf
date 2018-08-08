#!/bin/bash
#SBATCH --job-name=joint_dl_crf
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --output="jigsaws_joint_dl_crf_suturing_louo-%A_%a.out"
#SBATCH --error="jigsaws_joint_dl_crf_suturing_louo-%A_%a.err"
#SBATCH --array=0
#SBATCH --partition=shared
##SBATCH --gres=gpu:1
#SBATCH --ntasks=8
##SBATCH --cpus-per-task=6
#SBATCH --mail-type=end
#SBATCH --mail-user=ippatia2@gmail.com

datasets_dir='/home-2/emavrou1@jhu.edu/work/ar_video_datasets/'
output_dir="/home-2/emavrou1@jhu.edu/scratch/experiments/m2cai15/experiments_${dataset_name}_${model_name}_${suffix}"
output_csv_file="/home-2/emavrou1@jhu.edu/results_log_${dataset_name}_${model_name}_${suffix}.csv"
code_path='/home-2/emavrou1@jhu.edu/repos/wacv18_joint_dl_crf'
spams_matlab_path='/home-2/emavrou1@jhu.edu/work/ar_libs'
configs_path='/home-2/emavrou1@jhu.edu/repos/wacv18_joint_dl_crf/configuration_scripts'
module load matlab/R2017a

batch_id=$(($SLURM_ARRAY_TASK_ID + 1))
nb_batches=3
dataset_name='JIGSAWS'
model_name='joint_dl_crf'
suffix='exps_louo_final'

echo $batch_id
echo $nb_batches
echo $output_dir
echo $output_csv_file
echo $code_path
echo $spams_matlab_path

cd $configs_path
matlab -nodesktop -nodisplay nosplash -r "try exps_jigsaws_joint_dl_crf_suturing_louo($batch_id, $nb_batches, '$dataset_name', '$model_name', '$output_dir', '$output_csv_file', '$code_path', '$spams_matlab_path','$datasets_dir');catch; end; quit"

