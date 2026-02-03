from flask import Flask, render_template, request, redirect, send_from_directory, url_for, session
import numpy as np
import json
import uuid
import tensorflow as tf
import os

app = Flask(__name__)

app.secret_key = '2a3f8b9e6c1d7a5b3e9c4f0a1b8d7e6f9c2a5b1d8e7f6c3'

model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    result_data = session.pop('result_data', None)
    
    if result_data:
        return render_template('home.html', 
                               result=True, 
                               prediction=result_data['prediction'],
                               imagepath=result_data['imagepath'])
    else:
        return render_template('home.html', result=False)

def extract_features(image):
    image = tf.keras.utils.load_img(image,target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
    
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        save_path = f'{temp_name}_{image.filename}' 
        
        image.save(save_path)
    
        prediction = model_predict(f'./{save_path}') 
       
        image_url_path = f'/{save_path}'
    
        session['result_data'] = {
            'prediction': prediction,
            'imagepath': image_url_path 
        }
        return redirect(url_for('home') + '#result')
    else:
        return redirect('/')
    
if __name__ == "__main__":
    app.run(debug=True)