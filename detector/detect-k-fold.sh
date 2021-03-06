python detect.py \
	--weights runs/train/exp28/weights/best.pt \
	--source data/garbage_04/fold_1/images/val \
	--imgsz 640 \
	--conf-thres 0.25 \
	--iou-thres 0.45 \
	--device 0 \
	--save-txt \
	--save-conf \
    --name fold_1 \
	--nosave 
python detect.py \
	--weights runs/train/exp29/weights/best.pt \
	--source data/garbage_04/fold_2/images/val \
	--imgsz 640 \
	--conf-thres 0.25 \
	--iou-thres 0.45 \
	--device 0 \
	--save-txt \
	--save-conf \
    --name fold_2 \
	--nosave 
python detect.py \
	--weights runs/train/exp30/weights/best.pt \
	--source data/garbage_04/fold_3/images/val \
	--imgsz 640 \
	--conf-thres 0.25 \
	--iou-thres 0.45 \
	--device 0 \
	--save-txt \
	--save-conf \
    --name fold_3 \
	--nosave 
python detect.py \
	--weights runs/train/exp31/weights/best.pt \
	--source data/garbage_04/fold_4/images/val \
	--imgsz 640 \
	--conf-thres 0.25 \
	--iou-thres 0.45 \
	--device 0 \
	--save-txt \
	--save-conf \
    --name fold_4 \
	--nosave 
python detect.py \
	--weights runs/train/exp32/weights/best.pt \
	--source data/garbage_04/fold_5/images/val \
	--imgsz 640 \
	--conf-thres 0.25 \
	--iou-thres 0.45 \
	--device 0 \
	--save-txt \
	--save-conf \
    --name fold_5 \
	--nosave 