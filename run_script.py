import subprocess

# Define the model sizes you want to iterate over
model_sizes = ["tiny", "small", "medium"]

# Common command arguments
common_args = [
    "python", "plot_train.py",
    "--batch_size", "-1",
    "--num_workers", "16",
    "--prior_model_name", "resnet18_cifar100",
    "--num_epochs", "10",
    "--draw", "False",
    "--dataset", "cifar100",
    "--alpha_weight", "1",
    "--prior_token_position", "all"
]

# Iterate over the model sizes and run the command
for size in model_sizes:
    args = common_args.copy()
    args.insert(2, "--model_size")
    args.insert(3, size)
    subprocess.run(args)

    # Optional: print the command being executed for clarity
    print("Running command:", " ".join(args))
