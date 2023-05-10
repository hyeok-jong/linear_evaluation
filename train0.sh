

# Url error "stanfordcars"
datasets=("cifar10" "cifar100" "food101" "aircraft" "dtd" "pet")
methods=("linear" "finetune")
for method in "${methods[@]}"
do
  for dataset in "${datasets[@]}"
  do
    echo "Running with --dataset $dataset and --method $method"
    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --dataset $dataset \
    --method $method
  done
done

