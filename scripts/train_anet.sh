#!/bin/bash
#================================================================
#   Don't go gently into that good night. 
#   
#   author: klaus
#   description: 
#
#================================================================
set -ex

# 1.a.i
workdir=workdir/anet/1.a.i
config=configs/trainval/anet/1.a.i.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/anet/1.a.i
config=configs/trainval/anet/1.a.i.py
#epoch=900
for epoch in 200 100; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --dataset anet \
        --out $workdir/results_e$epoch-chunk.pkl
done


# 1.a.ii
workdir=workdir/anet/1.a.ii
config=configs/trainval/anet/1.a.ii.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/anet/1.a.ii
config=configs/trainval/anet/1.a.ii.py
epoch=50
# for epoch in 200 100; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --dataset anet \
        --out $workdir/results_e$epoch-chunk.pkl
done
