# !bin/bash
# for iemocap 4 way
# python3 train.py --dataset "iemocap" --from_begin --architecture "single_global_node_classifier" --epochs 50 --device cpu

# change seed
python3 train.py --dataset "iemocap" --from_begin --architecture "single_global_node" --epochs 50 --device cpu --seed 1027
# for iemocap 6 way
# python eval.py --dataset="iemocap" --modalities="atv"

