# train teacher
python -u main.py --dataset=gowalla --backbone=bpr --train_teacher
python -u main.py --dataset=gowalla --backbone=lightgcn --train_teacher

python -u main.py --dataset=foursquare --backbone=bpr --train_teacher
python -u main.py --dataset=foursquare --backbone=lightgcn --train_teacher

python -u main.py --dataset=citeulike --backbone=bpr --train_teacher
python -u main.py --dataset=citeulike --backbone=lightgcn --train_teacher

# KD
python -u main.py --dataset=gowalla --backbone=bpr --model=rrd

python -u main.py --dataset=foursquare --backbone=bpr --model=de
python -u main.py --dataset=foursquare --backbone=bpr --model=cpd
python -u main.py --dataset=foursquare --backbone=bpr --model=nkd

python -u main.py --dataset=citeulike --backbone=bpr --model=de
