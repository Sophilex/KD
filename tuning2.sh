# python main.py --dataset=citeulike --backbone=lightgcn --model=mrrd --gpu_id=1 --cfg mrrd_beta=0.2
# python main.py --dataset=citeulike --backbone=lightgcn --model=mrrd --gpu_id=1 --cfg mrrd_beta=0.3
# python main.py --dataset=citeulike --backbone=lightgcn --model=mrrd --gpu_id=1 --cfg mrrd_beta=0.4
# python main.py --dataset=citeulike --backbone=lightgcn --model=mrrd --gpu_id=1 --cfg mrrd_beta=0.5
# python main.py --dataset=citeulike --backbone=lightgcn --model=mrrd --gpu_id=1 --cfg mrrd_beta=0.6
# python main.py --dataset=citeulike --backbone=lightgcn --model=mrrd --gpu_id=1 --cfg mrrd_beta=0.7
# python main.py --dataset=citeulike --backbone=lightgcn --model=mrrd --gpu_id=1 --cfg mrrd_beta=0.8
# python main.py --dataset=citeulike --backbone=lightgcn --model=mrrd --gpu_id=1 --cfg mrrd_beta=0.9

python main.py --dataset=citeulike --backbone=lightgcn --model=dcdvar --cfg dcd_negx=0.1
python main.py --dataset=citeulike --backbone=lightgcn --model=dcdvar --cfg dcd_negx=0.5
python main.py --dataset=citeulike --backbone=lightgcn --model=dcdvar --cfg dcd_negx=1.5
python main.py --dataset=citeulike --backbone=lightgcn --model=dcdvar --cfg dcd_negx=2
python main.py --dataset=citeulike --backbone=lightgcn --model=dcdvar --cfg dcd_negx=5
