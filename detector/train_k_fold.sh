python -m torch.distributed.launch --nproc_per_node 4 \
train.py \
	--weights yolov5l.pt \
	--hyp data/hyps/hyp.scratch.yaml \
	--data data/k_fold/Garbage_5_fold_01.yaml \
	--batch-size 256 \
	--imgsz 640 \
	--device 0,1,2,3 \
	--epochs 100 
python -m torch.distributed.launch --nproc_per_node 4 \
train.py \
	--weights yolov5l.pt \
	--hyp data/hyps/hyp.scratch.yaml \
	--data data/k_fold/Garbage_5_fold_02.yaml \
	--batch-size 256 \
	--imgsz 640 \
	--device 0,1,2,3 \
	--epochs 100 
python -m torch.distributed.launch --nproc_per_node 4 \
train.py \
	--weights yolov5l.pt \
	--hyp data/hyps/hyp.scratch.yaml \
	--data data/k_fold/Garbage_5_fold_03.yaml \
	--batch-size 256 \
	--imgsz 640 \
	--device 0,1,2,3 \
	--epochs 100 
python -m torch.distributed.launch --nproc_per_node 4 \
train.py \
	--weights yolov5l.pt \
	--hyp data/hyps/hyp.scratch.yaml \
	--data data/k_fold/Garbage_5_fold_04.yaml \
	--batch-size 256 \
	--imgsz 640 \
	--device 0,1,2,3 \
	--epochs 100 
python -m torch.distributed.launch --nproc_per_node 4 \
train.py \
	--weights yolov5l.pt \
	--hyp data/hyps/hyp.scratch.yaml \
	--data data/k_fold/Garbage_5_fold_05.yaml \
	--batch-size 256 \
	--imgsz 640 \
	--device 0,1,2,3 \
	--epochs 100 
