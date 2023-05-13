

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--dataset stl10 \
--method linear \
--exper GenSCL \
--model_path ./trained_models/our_genscl.pth.tar

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--dataset stl10 \
--method linear \
--exper SupCon \
--model_path ./trained_models/our_supcon.pth.tar

CUDA_VISIBLE_DEVICES=2 \
python train.py \
--dataset food101 \
--method linear \
--exper GenSCL \
--model_path ./trained_models/our_genscl.pth.tar

CUDA_VISIBLE_DEVICES=3 \
python train.py \
--dataset food101 \
--method linear \
--exper SupCon \
--model_path ./trained_models/our_supcon.pth.tar


CUDA_VISIBLE_DEVICES=4 \
python train.py \
--dataset dtd \
--method linear \
--exper GenSCL \
--model_path ./trained_models/our_genscl.pth.tar

CUDA_VISIBLE_DEVICES=5 \
python train.py \
--dataset dtd \
--method linear \
--exper SupCon \
--model_path ./trained_models/our_supcon.pth.tar


CUDA_VISIBLE_DEVICES=6 \
python train.py \
--dataset flowers \
--method linear \
--exper GenSCL \
--model_path ./trained_models/our_genscl.pth.tar

CUDA_VISIBLE_DEVICES=7 \
python train.py \
--dataset flowers \
--method linear \
--exper SupCon \
--model_path ./trained_models/our_supcon.pth.tar


CUDA_VISIBLE_DEVICES=8 \
python train.py \
--dataset pet \
--method linear \
--exper GenSCL \
--model_path ./trained_models/our_genscl.pth.tar

CUDA_VISIBLE_DEVICES=9 \
python train.py \
--dataset pet \
--method linear \
--exper SupCon \
--model_path ./trained_models/our_supcon.pth.tar













CUDA_VISIBLE_DEVICES=0 \
python train.py \
--dataset aircraft \
--method linear \
--exper GenSCL \
--model_path ./trained_models/our_genscl.pth.tar

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--dataset aircraft \
--method linear \
--exper SupCon \
--model_path ./trained_models/our_supcon.pth.tar





CUDA_VISIBLE_DEVICES=2 \
python train.py \
--dataset dtd \
--method linear \
--exper GenSCL \
--model_path ./trained_models/our_genscl.pth.tar

CUDA_VISIBLE_DEVICES=3 \
python train.py \
--dataset dtd \
--method linear \
--exper SupCon \
--model_path ./trained_models/our_supcon.pth.tar



CUDA_VISIBLE_DEVICES=4 \
python train.py \
--dataset sun \
--method linear \
--exper GenSCL \
--model_path ./trained_models/our_genscl.pth.tar

CUDA_VISIBLE_DEVICES=5 \
python train.py \
--dataset sun \
--method linear \
--exper SupCon \
--model_path ./trained_models/our_supcon.pth.tar











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




