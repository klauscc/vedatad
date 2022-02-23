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

# 1.a.iii
workdir=workdir/anet/1.a.iii
config=configs/trainval/anet/1.a.iii.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/anet/1.a.iii
config=configs/trainval/anet/1.a.iii.py
# epoch=50
for epoch in 10 20 25 30 35 40 50; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --dataset anet \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 1.b.i
workdir=workdir/anet/1.b.i
config=configs/trainval/anet/1.b.i.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/anet/1.b.i
config=configs/trainval/anet/1.b.i.py
epoch=30
# for epoch in 200 100; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --dataset anet \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 1.c.i
workdir=workdir/anet/1.c.i
config=configs/trainval/anet/1.c.i.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/anet/1.c.i
config=configs/trainval/anet/1.c.i.py
# epoch=30
for epoch in 10 15 20 25 30; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --dataset anet \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 1.c.ii
workdir=workdir/anet/1.c.ii
config=configs/trainval/anet/1.c.ii.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/anet/1.c.ii
config=configs/trainval/anet/1.c.ii.py
# epoch=30
for epoch in 10 15 20 25 30; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --dataset anet \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 1.d.i
workdir=workdir/anet/1.d.i
config=configs/trainval/anet/1.d.i.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/anet/1.d.i
config=configs/trainval/anet/1.d.i.py
# epoch=30
for epoch in 10 15 20 25 30; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --dataset anet \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 1.e.i
workdir=workdir/anet/1.e.i
config=configs/trainval/anet/1.e.i.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/anet/1.e.i
config=configs/trainval/anet/1.e.i.py
# epoch=15
for epoch in 10 15 20 25 30; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --dataset anet \
        --out $workdir/results_e$epoch-chunk.pkl
done
