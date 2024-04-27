export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONPATH=$(dirname "$0")

python main.py \
--dataset_dir /root/autodl-tmp/ffhq_small \
--pretrained_path /root/autodl-tmp/pretrained_models \
--save_path /root/autodl-tmp/output \
--batch_size 8 \
--epochs 10 \
--show_interval 1 \
--lr 0.05 \
--wd 0.0005 \
--eps 1e-8 \
--eta_min 0 \
--lr_scheduler \
--seed 0 \
--face_parsing_lambda 0.1 \
--id_lambda 0.1 \
--l2_lambda 1.0 \
--lpips_lambda 0.8 \
--norm_lambda 0.01
