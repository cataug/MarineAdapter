
set -e
source /home/tahiti/Malashin_Projects/.venv_a100/bin/activate
cd /home/tahiti/MARINE_DATASETS

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[START] 2026-06-11 20:38:14"
echo "JOB: marine_adapter_b005_s44_ad256"
echo "CMD: /home/tahiti/Malashin_Projects/.venv_a100/bin/python /home/tahiti/MARINE_DATASETS/run_marine_adapter_experiments.py --out_dir /home/tahiti/MARINE_DATASETS/MARINE_RESULTS_DYNAMIC --epochs 10 --batch_size 16 --img_size 224 --num_workers 0 --max_classes 30 --max_images_per_class 300 --methods marine_adapter --budgets 0.05 --seeds 44 --adapter_dim 256"
echo

/home/tahiti/Malashin_Projects/.venv_a100/bin/python /home/tahiti/MARINE_DATASETS/run_marine_adapter_experiments.py --out_dir /home/tahiti/MARINE_DATASETS/MARINE_RESULTS_DYNAMIC --epochs 10 --batch_size 16 --img_size 224 --num_workers 0 --max_classes 30 --max_images_per_class 300 --methods marine_adapter --budgets 0.05 --seeds 44 --adapter_dim 256

echo
echo "[DONE] $(date '+%Y-%m-%d %H:%M:%S')"
