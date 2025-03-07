# python main.py --dataset=gowalla --backbone=simplex --model=rrdvk --gpu_id=2 --cfg rrd_neg=2
# python main.py --dataset=citeulike --backbone=bpr --model=rrdvk --gpu_id=2 --cfg rrd_neg=3
# python main.py --dataset=citeulike --backbone=bpr --model=rrdvk --gpu_id=2 --cfg rrd_neg=4
# python main.py --dataset=citeulike --backbone=bpr --model=rrdvk --gpu_id=2 --cfg rrd_neg=5
# python main.py --dataset=citeulike --backbone=bpr --model=rrdvk --gpu_id=3 --cfg rrd_neg=10

# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg rrd_neg=0
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg rrd_neg=1
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg rrd_neg=20
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg rrd_neg=30
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg rrd_neg=50
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg rrd_neg=100

# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg calu_len=3
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg calu_len=4
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg calu_len=10
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg calu_len=20
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=1 --cfg calu_len=50

python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=2 --cfg mode="val_diff" alpha=0.01
python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=2 --cfg mode="val_diff" alpha=0.1
python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=2 --cfg mode="val_diff" alpha=1
python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=2 --cfg mode="val_diff" alpha=10
python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=2 --cfg mode="val_diff" alpha=20
python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvk --gpu_id=2 --cfg mode="val_diff" alpha=100
# python main.py --dataset=gowalla --backbone=simplex --model=rrdvk --cfg rrd_neg=3
# python main.py --dataset=gowalla --backbone=simplex --model=rrdvk --cfg rrd_neg=5
# python main.py --dataset=gowalla --backbone=simplex --model=rrdvk --cfg rrd_neg=10

