import streamlit as st
from garbageNetDeploy import garbageNet
from PIL import Image


INT_TO_LABEL = {0:"종이", 1:"종이팩", 2:"알류미늄캔", 3:"유리", 4:"페트", 5:"플라스틱", 6:"비닐"}
BBOX_MODEL_PATH = './garbage_bbox_jit.pth'
CLS_MODEL_PATH = './garbage_cls_jit.pth'

BBOX_IMAGE_SIZE = 456
BBOX_MARGIN = 1/8
BBOX_DEVICE = 'cpu'
CLS_DEVICE = 'cpu'
CLS_IMAGE_SIZE = 456
CLS_ADVPROP = True

def run():

    st.title("생활폐기물 인식 데모")
    st.text("생활폐기물 사진을 업로드하시면, 어떤 종류의 폐기물인지 분류해 드립니다!")

    Classifier = garbageNet(BBOX_MODEL_PATH,CLS_MODEL_PATH,BBOX_IMAGE_SIZE,BBOX_MARGIN,BBOX_DEVICE,CLS_DEVICE, CLS_IMAGE_SIZE, CLS_ADVPROP)
    
    uploaded_file = st.file_uploader("생활폐기물 사진을 선택해주세요", type=["jpg","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='업로드된 생활폐기물 사진.', use_column_width=True)
        st.write("\n")
        #st.write("분류중...")
        with st.spinner('분류중 ...'):
            labels, scores = Classifier.run(uploaded_file)
            st.success('작업이 완료되었습니다')
        
        

        message = ', '.join(['{}({:.4f})'.format(INT_TO_LABEL[i], scores[i]) for i,val in enumerate(labels) if int(val)==1])
        if len(message)==0:
            st.header("알고리즘 인식결과:폐기물이 인식되지 않았습니다.")
        else:
            st.header("알고리즘 인식결과: {}".format(message))

if __name__ == '__main__':
	run()

