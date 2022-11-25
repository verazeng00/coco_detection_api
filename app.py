import io
from operator import truediv
import os
import json
from PIL import Image
import cv2 
import numpy as np 
from flask import Flask, jsonify, url_for, render_template, request, redirect


from utils.utils import BaseEngine

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER


class Predictor(BaseEngine):
    def __init__(self, engine_path , imgsz=(640,640)):
        super(Predictor, self).__init__(engine_path)
        self.imgsz = imgsz # your model infer image size
        self.n_classes = 80  # your model classes



def get_model():
    model = Predictor(engine_path= 'models/coco/yolov7-tiny.trt')
    print('model', model)
    return model 


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
            
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))

        opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        model = get_model()
        pred_img = model.inference(opencv_img, conf=0.1, end2end=True)
        filename = 'image_out.jpg'
        cv2.imwrite('static/' + filename , pred_img)
        
        
        return render_template('result.html',result_image = filename,model_name = 'yolov7-tiny')

    return render_template('index.html')
    
@app.route('/video', methods=['GET', 'POST'])
def handle_video():
    # pred.detect_video(video, conf=0.1, end2end=args.end2end) # set 0 use a webcam

    pass

@app.route('/webcam', methods=['GET', 'POST'])
def web_cam():
    # some code to be implemented later
    pass


if __name__ == '__main__':
     app.run(host='0.0.0.0', port = 5050, debug=False)