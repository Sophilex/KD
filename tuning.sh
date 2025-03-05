# python main.py --dataset=citeulike --backbone=jgcf --model=dcdoptim4 --cfg  dcd_y=20
# python main.py --dataset=citeulike --backbone=jgcf --model=dcdoptim4 --cfg  dcd_y=15
# python main.py --dataset=citeulike --backbone=jgcf --model=dcdoptim4 --cfg  dcd_y=10
# python main.py --dataset=citeulike --backbone=jgcf --model=dcdoptim4 --cfg  dcd_y=5
# python main.py --dataset=citeulike --backbone=jgcf --model=dcdoptim4 --cfg  dcd_y=1
# python main.py --dataset=citeulike --backbone=jgcf --model=dcdoptim4 --cfg  dcd_y=-5
# python main.py --dataset=citeulike --backbone=jgcf --model=dcdoptim4 --cfg  dcd_y=-10
# python main.py --dataset=citeulike --backbone=jgcf --model=dcdoptim4 --cfg  dcd_y=-15
# python main.py --dataset=citeulike --backbone=jgcf --model=dcdoptim4 --cfg  dcd_y=-20

# python main.py --dataset=citeulike --backbone=jgcf --model=rrd
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=0.1
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=0.3
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=0.5
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=0.7
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=0.9
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=1
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=1.1
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=1.3
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=1.6
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd --gpu_id=3 --cfg rrd_unselected=1.9

# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=0.1
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=3 --train_teacher

# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=0.1
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=0.3
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=0.5
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=0.7
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=0.9
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=1
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=1.1
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=1.3
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=1.6
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --gpu_id=1 --cfg rrd_unselected=1.9

# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd --cfg  draw_student=True
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrd
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvar --cfg rrd_neg=1.5  draw_student=True
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvar --cfg rrd_neg=2
# python main.py --dataset=citeulike --backbone=lightgcn --model=rrdvar --cfg rrd_neg=5



# citeulike
# python main.py --dataset=citeulike --backbone=bpr --model=rrdvar
# python main.py --dataset=citeulike --backbone=bpr --model=rrd
# python main.py --dataset=citeulike --backbone=bpr --model=dcd

# python main.py --dataset=citeulike --backbone=simplex --model=rrdvar
# python main.py --dataset=citeulike --backbone=simplex --model=rrd
# python main.py --dataset=citeulike --backbone=simplex --model=dcd

# python main.py --dataset=citeulike --backbone=jgcf --model=rrdvar
# python main.py --dataset=citeulike --backbone=jgcf --model=rrd
# python main.py --dataset=citeulike --backbone=jgcf --model=dcd


# gowalla
# python main.py --dataset=gowalla --backbone=bpr --model=rrdvar
# python main.py --dataset=gowalla --backbone=bpr --model=rrd
# python main.py --dataset=gowalla --backbone=bpr --model=dcd

# python main.py --dataset=gowalla --backbone=simplex --model=rrdvar
# python main.py --dataset=gowalla --backbone=simplex --model=rrd
# python main.py --dataset=gowalla --backbone=simplex --model=dcd

# python main.py --dataset=gowalla --backbone=jgcf --model=rrdvar
# python main.py --dataset=gowalla --backbone=jgcf --model=rrd
# python main.py --dataset=gowalla --backbone=jgcf --model=dcd


# yelp
# python main.py --dataset=yelp --backbone=bpr --model=rrdvar
# python main.py --dataset=yelp --backbone=bpr --model=rrd
# python main.py --dataset=yelp --backbone=bpr --model=dcd

# python main.py --dataset=yelp --backbone=simplex --model=rrdvar
# python main.py --dataset=yelp --backbone=simplex --model=rrd
# python main.py --dataset=yelp --backbone=simplex --model=dcd

# python main.py --dataset=yelp --backbone=jgcf --model=rrdvar
# python main.py --dataset=yelp --backbone=jgcf --model=rrd
# python main.py --dataset=yelp --backbone=jgcf --model=dcd


python main.py --dataset=gowalla --backbone=lightgcn --model=rrdvk
python main.py --dataset=gowalla --backbone=lightgcn --model=rrdvk --cfg alpha=0.5
python main.py --dataset=gowalla --backbone=lightgcn --model=rrdvk --cfg alpha=0.1
python main.py --dataset=gowalla --backbone=lightgcn --model=rrdvk --cfg alpha=1.5
python main.py --dataset=gowalla --backbone=lightgcn --model=rrdvk --cfg alpha=2










