
# 2.a.i.1
expid=2.a.i.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=2.a.i.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
for epoch in 400 500 600 700 800 900 1000; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 2.a.ii.1
expid=2.a.ii.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=2.a.ii.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
# for epoch in 400 500 600 700 800 900 1000; do
for epoch in 600 700 800 900 1000; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 2.a.iii.2
expid=2.a.iii.2
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=2.a.iii.2
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
for epoch in 400 500 600 700 800 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 2.a.iii.1
expid=2.a.iii.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
expid=2.a.iii.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
for epoch in 400 500 700 800 900 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done


# 1.c.iv.3.a
expid=1.c.iv.3.a
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=1.c.iv.3.a
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
epoch=500
# for epoch in 600 700 800 900 400 500 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 1.c.iv.3.b
expid=1.c.iv.3.b
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=1.c.iv.3.b
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
for epoch in 600 700 800 900 400 500 1000 1100 1200; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 6.a.i.1
expid=6.a.i.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=6.a.i.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
for epoch in 600 700 800 900 400 500 1000; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 6.b.i.2
expid=6.b.i.2
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=6.b.i.2
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
for epoch in 600 700 800 900 400 500 1000; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 6.b.i.3
expid=6.b.i.3
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=6.b.i.3
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
for epoch in 600 700 800 900 400 500 1000; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 6.b.ii.1
expid=6.b.ii.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir
# Testing chunk
expid=6.b.ii.1
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
for epoch in 600 700  400 500 800 900 1000; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

# 6.c.i
expid=6.c.i
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
tools/dist_trainval.sh $config "4,5,6,7" --workdir $workdir
# Testing chunk
expid=6.c.i
workdir=workdir/eccv2022/$expid
config=configs/trainval/daotad_eccv2022/$expid.py
#epoch=900
for epoch in 600 700 800 900 400 500 1000; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done
