# Check if 6 or 7 arguments are provided
if [ "$#" -lt 6 ] || [ "$#" -gt 7 ]; then
    echo "Usage: $0 <number of times to run> <dataset> <architecture> <modalities> <device> <global_aggregate_method> [<wandb_run>]"
    exit 1
fi

# Assign arguments to variables
times_to_run=$1
dataset=$2
architecture=$3
modalities=$4
device=$5
global_aggregate_method=$6
wandb_run=${7:-"default"}  # Default value if $7 is not provided

# Check if the number of times to run is a valid number
if ! [[ $times_to_run =~ ^[0-9]+$ ]]; then
    echo "Error: First argument must be a positive integer."
    exit 1
fi

# Execute the Python script the specified number of times
for ((i=1; i<=times_to_run; i++))
do
    echo "Execution $i of training"
    python3 train.py --dataset $dataset --from_begin --architecture $architecture --modalities $modalities --epochs 50 --wandb --wandb_run $wandb_run --device $device --global_aggregate_method $global_aggregate_method --learning_rate 0.000101
done
