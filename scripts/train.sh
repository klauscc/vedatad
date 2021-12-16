#!/bin/bash
#================================================================
#   Don't go gently into that good night. 
#   
#   author: klaus
#   description: 
#
#================================================================
set -ex

<<COMMENT
# 1.a.iii
workdir=workdir/1.a.iii
tools/dist_trainval.sh configs/trainval/daotad/daotad_vswin_t_e700_thumos14_rgb.py "0,1,2,3" --workdir $workdir
epoch=700
python tools/test.py configs/trainval/daotad/daotad_vswin_t_e700_thumos14_rgb.py $workdir/epoch_${epoch}_weights.pth \
    --out $workdir/results_e$epoch.pkl
# Testing whole
workdir=workdir/1.a.iii
config=configs/trainval/daotad/1.a.iii.py
for epoch in 900 800 700 1000 1100 1200; do
    echo Epoch: $epoch. Inferencing...
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-whole.pkl
done
COMMENT

## 1.b.ii
#workdir=workdir/1.b.ii
#tools/dist_trainval.sh configs/trainval/daotad/daotad_vswin_t_e700_thumos14_rgb_224x224.py "0,1,2,3,4,5,6,7" --workdir $workdir
###test
#epoch=700
#CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/trainval/daotad/daotad_vswin_t_e700_thumos14_rgb_224x224.py $workdir/epoch_${epoch}_weights.pth \
#    --out $workdir/results_e$epoch.pkl

## 1.c.ii
#workdir=workdir/1.c.ii
#epoch=700
#config=configs/trainval/daotad/1.c.ii.py
#tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
#python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
#    --out $workdir/results_e$epoch.pkl

## 1.c.iii
#workdir=workdir/1.c.iii
#epoch=700
#config=configs/trainval/daotad/1.c.iii.py
#tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
#python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
#    --out $workdir/results_e$epoch.pkl

## 1.d.iii
#workdir=workdir/1.d.iii
#epoch=700
#config=configs/trainval/daotad/1.d.iii.py
#tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
#python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
#    --out $workdir/results_e$epoch.pkl

## 1.d.ii
#workdir=workdir/1.d.ii
#epoch=700
#config=configs/trainval/daotad/1.d.ii.py
#tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
#python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
#    --out $workdir/results_e$epoch.pkl

## 1.e.i
#workdir=workdir/1.e.i
#epoch=700
#config=configs/trainval/daotad/1.e.i.py
#tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
#python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
#    --out $workdir/results_e$epoch.pkl

## 1.e.iii
#workdir=workdir/1.e.iii
#epoch=700
#config=configs/trainval/daotad/1.e.iii.py
#tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
#python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
#    --out $workdir/results_e$epoch.pkl

<<COMMENT
# 2.a.ii
workdir=workdir/2.a.ii
config=configs/trainval/daotad/2.a.ii.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Tesing chunk
workdir=workdir/2.a.ii
config=configs/trainval/daotad/2.a.ii.py
for epoch in 700 800 900 1000 1100 1200; do
    echo Epoch: $epoch. Inferencing...
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
# Testing whole
workdir=workdir/2.a.ii
config=configs/trainval/daotad/1.a.iii.py
for epoch in 700 800 900 1000 1100 1200; do
    echo Epoch: $epoch. Inferencing...
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-whole.pkl
done
COMMENT

<<COMMENT
# 2.b.ii
workdir=workdir/2.b.ii_run2 
config=configs/trainval/daotad/2.b.ii.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/2.b.ii_run2 
config=configs/trainval/daotad/2.b.ii.py
for epoch in 700 800 900 1000 1100 1200; do
    echo Epoch: $epoch. Inferencing...
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
# Testing whole
workdir=workdir/2.b.ii
config=configs/trainval/daotad/1.a.iii.py
for epoch in 700 800 900 1000 1100 1200; do
    echo Epoch: $epoch. Inferencing...
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-whole.pkl
done
COMMENT

<<COMMENT
# 2.b.ii.2
workdir=workdir/2.b.ii.2_v2 
config=configs/trainval/daotad/2.b.ii.2_v2.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/2.b.ii.2_v2 
config=configs/trainval/daotad/2.b.ii.2_v2.py
for epoch in 700 800 900 1000 1100 1200; do
    echo Epoch: $epoch. Inferencing...
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

#<<COMMENT
# 2.b.ii_v2
workdir=workdir/2.b.ii_v2
config=configs/trainval/daotad/2.b.ii_v2.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/2.b.ii_v2
config=configs/trainval/daotad/2.b.ii_v2.py
epoch=900
python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
    --out $workdir/results_e$epoch-chunk.pkl
#COMMENT

<<COMMENT
# 2.b.ii.3
workdir=workdir/2.b.ii.3-run1
config=configs/trainval/daotad/2.b.ii.3.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/2.b.ii.3-run1
config=configs/trainval/daotad/2.b.ii.3.py
#epoch=900
for epoch in 700 800 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 2.b.ii.4
expid=2.b.ii.4
workdir=workdir/${expid}
config=configs/trainval/daotad/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/$expid
config=configs/trainval/daotad/$expid.py
#epoch=900
for epoch in 700 800 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 2.b.ii.5
expid=2.b.ii.5
workdir=workdir/${expid}
config=configs/trainval/daotad/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/$expid
config=configs/trainval/daotad/$expid.py
#epoch=900
for epoch in 100 200 300 400 500 600 700 800 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 2.b.iii
workdir=workdir/2.b.iii
config=configs/trainval/daotad/2.b.iii.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/2.b.iii
config=configs/trainval/daotad/2.b.iii.py
epoch=900
python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
    --out $workdir/results_e$epoch-chunk.pkl
COMMENT

<<COMMENT
# 2.b.iv
workdir=workdir/2.b.iv
config=configs/trainval/daotad/2.b.iv.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/2.b.iv
config=configs/trainval/daotad/2.b.iv.py
epoch=700
python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
    --out $workdir/results_e$epoch-chunk.pkl
COMMENT

<<COMMENT
# 2.b.vi
workdir=workdir/2.b.vi
config=configs/trainval/daotad/2.b.vi.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/2.b.vi
config=configs/trainval/daotad/2.b.vi.py
epoch=500
#for epoch in 100 200 300 400 500 600 700 800 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 2.b.vii
workdir=workdir/2.b.vii
config=configs/trainval/daotad/2.b.vii.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/2.b.vii
config=configs/trainval/daotad/2.b.vii.py
epoch=600
#for epoch in 100 200 300 400 500 600 700 800 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 2.b.viii
workdir=workdir/2.b.viii
config=configs/trainval/daotad/2.b.viii.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/2.b.viii
config=configs/trainval/daotad/2.b.viii.py
#epoch=600
for epoch in 300 600 800 700 400 500 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 3.a.i
mkdir -p data/tmp/thumos14/memory_mechanism/3.a.i
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/thumos14/memory_mechanism/3.a.i/
workdir=workdir/3.a.i
config=configs/trainval/daotad/3.a.i.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/3.a.i
config=configs/trainval/daotad/3.a.i_eval.py
#epoch=700
for epoch in 300 600 800 900 1000 1100 1200 400 500; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 3.a.ii
mkdir -p data/tmp/thumos14/memory_mechanism/3.a.ii
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/thumos14/memory_mechanism/3.a.ii/
workdir=workdir/3.a.ii
config=configs/trainval/daotad/3.a.ii.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/3.a.ii
config=configs/trainval/daotad/3.a.ii_eval.py
#epoch=700
for epoch in 300 600 800 700 400 500 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 3.a.iii
mkdir -p data/tmp/thumos14/memory_mechanism/3.a.iii
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/thumos14/memory_mechanism/3.a.iii/
workdir=workdir/3.a.iii
config=configs/trainval/daotad/3.a.iii.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/3.a.iii
config=configs/trainval/daotad/3.a.iii_eval.py
#epoch=700
for epoch in 900 800 700 600 500 400; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 3.a.iv
workdir=workdir/3.a.iv
config=configs/trainval/daotad/3.a.iv.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/3.a.iv
config=configs/trainval/daotad/3.a.iv.py
#epoch=700
for epoch in 900 800 700 600 500 400; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT



<<COMMENT
# 3.b.i
mkdir -p data/tmp/thumos14/memory_mechanism/3.b.i
cp -r data/thumos14/memory_mechanism/feat_swinb_15fps_256x256_crop224x224 data/tmp/thumos14/memory_mechanism/3.b.i/
workdir=workdir/3.b.i
config=configs/trainval/daotad/3.b.i.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/3.b.i
config=configs/trainval/daotad/3.b.i_eval.py
#epoch=600
for epoch in 300 600 800 700 400 500 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 3.c.i
mkdir -p data/tmp/thumos14/memory_mechanism/3.c.i
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/thumos14/memory_mechanism/3.c.i/
workdir=workdir/3.c.i
config=configs/trainval/daotad/3.c.i.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/3.c.i
config=configs/trainval/daotad/3.c.i_eval.py
#epoch=700
for epoch in 300 600 800 900 1000 1100 1200 400 500; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 3.c.i.1
workdir=workdir/3.c.i.1
config=configs/trainval/daotad/3.c.i.1.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
workdir=workdir/3.c.i.1
config=configs/trainval/daotad/3.c.i.1.py
#epoch=700
for epoch in 600 800 900 300 1000 1100 1200 400 500; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

<<COMMENT
# 3.d.i
mkdir -p data/tmp/thumos14/memory_mechanism/3.d.i
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/thumos14/memory_mechanism/3.d.i/
workdir=workdir/3.d.i
config=configs/trainval/daotad/3.d.i.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
workdir=workdir/3.d.i
config=configs/trainval/daotad/3.d.i.py
#epoch=700
for epoch in 300 600 800 900 1000 1100 1200 400 500; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
COMMENT

