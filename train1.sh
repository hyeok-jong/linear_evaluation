datasets=("dtd" "pet" "caltech101" "flowers" "sun")
methods=("linear" "finetune")
for method in "${methods[@]}"
do
  for dataset in "${datasets[@]}"
  do
    echo "Running with --dataset $dataset and --method $method"
    CUDA_VISIBLE_DEVICES=1 \
    python train.py \
    --dataset $dataset \
    --method $method
  done
done