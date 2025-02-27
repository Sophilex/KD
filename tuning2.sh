# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=2
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=1
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=3
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=5
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=10

# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=2 rrd_extra=9000
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=10 rrd_extra=9000
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=20 rrd_extra=9000
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=30 rrd_extra=9000
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvk --cfg rrd_neg=40 rrd_extra=9000

python main.py --dataset=gowalla --backbone=lightgcn --train_teacher
# python main.py --dataset=citeulike --backbone=simplex --model=rrdvk





