FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update 
RUN apt-get install openjdk-11-jdk -y

COPY ./deploy/app.py ./app.py
COPY ./deploy/garbageNetDeploy.py ./garbageNetDeploy.py
COPY ./deploy/requirements.txt ./requirements.txt

RUN pip install -r ./requirements.txt

ADD https://dingbro-garbage.s3.ap-northeast-2.amazonaws.com/garbage_bbox_jit.pth ./garbage_bbox_jit.pth
ADD https://dingbro-garbage.s3.ap-northeast-2.amazonaws.com/garbage_cls_jit.pth ./garbage_cls_jit.pth

EXPOSE 6006

CMD ["streamlit", "run", "--server.port", "6006", "app.py"]
