
# model_name choices
# model_names=("resnet50-supcon" "regnet-y-16gf-swag-e2e-v1")
# model_names=("vit-b16-swag-e2e-v1" "vit-b-16" "vit-lp" "swin-t")
model_names=("resnet50-supcon" "vit-b-16" "vit-lp")
# model_names=("vit_cifar100")
# model_names=('resnet18_cifar100' 'vit_cifar100')

# id_data_name choices
# id_data_names=("imagenet1k" "imagenet1k-v2-a" "imagenet1k-v2-b" "imagenet1k-v2-c")
id_data_names=("imagenet1k")
# id_data_names=("cifar100")

# ood_data_name choices
ood_data_names=("inaturalist" "sun" "places" "textures" "openimage-o" "ssb_hard" "ninco")
# ood_data_names=('cifar10' 'textures' 'places' 'LSUN_C' 'iSUN' 'svhn')

# ood_detectors choices
# ood_detectors=("energy" "nnguide" "msp" "maxlogit" "vim" "ssd" "mahalanobis" "knn")
ood_detectors=("pvit")

scores=("cross_entropy" "KL" "JS" "dis")

# Convert array to string for the argument
detectors_arg="${ood_detectors[*]}"

# Iterate through all combinations
for model in "${model_names[@]}"; do
    for id_data in "${id_data_names[@]}"; do
        for ood_data in "${ood_data_names[@]}"; do
            for score in "${scores[@]}"; do
                # Running the main.py with the current combination
                # python main.py --model_name pvit --prior_model "$model" --id_data_name "$id_data" --ood_data_name "$ood_data" --ood_detectors pvit --batch_size 512 --num_workers 16 --pvit --score "$score"
                python main.py --model_name pvit --prior_model "$model" --id_data_name "$id_data" --ood_data_name "$ood_data" --ood_detectors pvit --batch_size 512 --num_workers 16 --pvit --score "$score" --seed 2 --run_ablation
                # python main.py --model_name pvit --train_data_name cifar100 --id_data_name cifar100 --prior_model "$model" --id_data_name "$id_data" --ood_data_name "$ood_data" --ood_detectors pvit --batch_size 512 --num_workers 16 --pvit --score "$score" --seed 0
            done
        done
    done
done

# python main.py --model_name pvit --id_data_name cifar100 --ood_data_name cifar10 --ood_detectors pvit --pvit --batch_size 512 --num_workers 1 --prior_model vit_cifar100 --score cross_entropy --seed 1

# python main.py --model_name pvit --id_data_name imagenet1k --ood_data_name inaturalist --ood_detectors pvit --batch_size 512 --num_workers 1 --prior_model vit-lp --pvit --score cross_entropy

# python main.py --model_name vit_pross --id_data_name imagenet1k --ood_data_name inaturalist --ood_detectors step --batch_size 512 --num_workers 1 --seed 1 --train