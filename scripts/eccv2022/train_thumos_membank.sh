#!/usr/bin/env sh

# 4.b.i
expid=4.b.i
mkdir -p data/tmp/eccv2022/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/eccv2022/thumos14/memory_mechanism/$expid/
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
expid=4.b.i
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
# epoch=900
for epoch in 600 700 800 900 500 1000 1100 400 300 1200 ; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 4.b.ii
expid=4.b.ii
mkdir -p data/tmp/eccv2022/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/eccv2022/thumos14/memory_mechanism/$expid/
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
expid=4.b.ii
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
# epoch=900
for epoch in 600 700 800 900 500 1000 1100 400 300 1200 ; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 4.b.iii
expid=4.b.iii
mkdir -p data/tmp/eccv2022/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/eccv2022/thumos14/memory_mechanism/$expid/
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=4.b.iii
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
# epoch=900
for epoch in 600 700 800 900 500 1000 1100 400 300 1200 ; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 4.b.iv
expid=4.b.iv
mkdir -p data/tmp/eccv2022/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/eccv2022/thumos14/memory_mechanism/$expid/
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=4.b.iv
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
# epoch=900
for epoch in 600 700 800 900 500 1000 1100 400 300 1200 ; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 4.b.v
expid=4.b.v
mkdir -p data/tmp/eccv2022/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112 data/tmp/eccv2022/thumos14/memory_mechanism/$expid/
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=4.b.v
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
# epoch=900
for epoch in 600 700 800 900 500 1000 1100 400 300 1200 ; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 6.a.i.2
expid=6.a.i.2
mkdir -p data/tmp/eccv2022/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swinb_15fps_256x256_crop224x224 data/tmp/eccv2022/thumos14/memory_mechanism/$expid/
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
expid=6.a.i.2
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
# epoch=900
for epoch in 600 700 900 800 500 1000 400 300; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 6.b.ii.2
expid=6.b.ii.2
mkdir -p data/tmp/eccv2022/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swinb_15fps_256x256_crop224x224 data/tmp/eccv2022/thumos14/memory_mechanism/$expid/
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
expid=6.b.ii.2
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
# epoch=900
for epoch in 600 700 800 900 500 1000 400 300; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
