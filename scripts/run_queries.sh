## TODOs
# * run again without resampling
# * distance with MFCC's? 

# set -e

# echo " === IDen === "
# python3 simulate_queries.py \
    # --name IDen_20q_0.35c \
    # --query_method IDen \
    # --oversample;


# echo " === Adaptive === "
# python3 simulate_queries.py \
#     --name Adaptive_IDiv_20q_0.35c \
#     --query_method Adaptive-IDiv \
#     --oversample;


echo " === Selective-IDiv === "
python3 simulate_queries.py \
    --name Selective-IDiv_20q_0.35c \
    --query_method Selective-IDiv \
    --oversample;


echo " === Embedding-IDiv === "
python3 simulate_queries.py \
    --name Embedding-IDiv_20q_0.35c \
    --query_method Embedding-IDiv \
    --oversample;


echo " === S-Coreset === "
python3 simulate_queries.py \
    --name S-Coreset_20q_0.35c \
    --query_method S-Coreset \
    --oversample;