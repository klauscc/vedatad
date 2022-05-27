

# r1.a
expid=r1.a
prefix="eccv2022/rebuttal"
mkdir -p data/tmp/$prefix/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swinb_15fps_256x256_crop224x224 data/tmp/$prefix/thumos14/memory_mechanism/$expid/
workdir=workdir/$prefix/$expid
config=configs/trainval/daotad_eccv2022/rebuttal/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=r1.a
workdir=workdir/$prefix/$expid
config=configs/trainval/daotad_eccv2022/rebuttal/$expid.py
# epoch=900
for epoch in 600 700 800 900 500 1000 400 300; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# r1.a.i
expid=r1.a.i
prefix="eccv2022/rebuttal"
mkdir -p data/tmp/$prefix/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swint_15fps_256x256_crop224x224 data/tmp/$prefix/thumos14/memory_mechanism/$expid/
workdir=workdir/$prefix/$expid
config=configs/trainval/daotad_eccv2022/rebuttal/$expid.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
expid=r1.a.i
prefix="eccv2022/rebuttal"
workdir=workdir/$prefix/$expid
config=configs/trainval/daotad_eccv2022/rebuttal/$expid.py
# epoch=900
for epoch in 600 700 800 500 900 1000 400 300; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
