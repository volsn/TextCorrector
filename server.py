import imutils
import cv2
import numpy as np
from flask import render_template, Flask, \
                request, make_response, send_file
import os
from converter import *
from werkzeug import secure_filename

app = Flask(__name__, static_url_path='/static/site', static_folder='templates')
DEBUG = True

app.config['UPLOAD_FOLDER'] = 'static'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        image = request.files['file']
        
        print(request.form)

        mode = request.form['mode']
        iterations = int(request.form['iterations'])
        ksize = int(request.form["ksize"])
        color = request.form['color']
        
        output = main(image, mode, ksize=ksize, iterations=iterations)
        cv2.imwrite('static/output.png', output)
        
        (H,W) = output.shape[:2]
        H /= W / 720
        W = 720
        
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'vector.svg')
        
        generate_svg(output, color=color)
        
        return send_file(full_filename)

def main(image, mode, ksize=5, iterations=3):
    
    image.save('temp.png')
    image = cv2.imread('temp.png')
    
    output = np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    letters, boxes = locate_letters(image)
    
    #cv2.imwrite('jui.png', letters[0])
    
    letters = preprocess_letters(letters)
    letters = affect_letters(letters, mode, ksize=ksize, iterations=iterations)
    
    for letter, box in zip(letters, boxes):
        
        (startY,endY,startX,endX) = box
        output[startY:endY, startX:endX][letter == 0] = 0
        
        
    return output


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
