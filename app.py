from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image as tfimage
from tensorflow.keras.applications.mobilenet import decode_predictions
from PIL import Image
import pickle
import numpy as np
import io

app = Flask(__name__)
cv = pickle.load(open("models/EmailClassifier/cv.pkl", "rb"))
clf = pickle.load(open("models/EmailClassifier/clf.pkl", "rb"))
MobileNet = pickle.load(open("models/DogClassifier/MobileNet.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

# get the prediction of the email
@app.route("/predictEmail", methods=["POST"])
def predictEmail():
    email_text = request.form.get('content')
    tokenized_email = cv.transform([email_text])
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email_text=email_text)

# get the prediction of thedog image
@app.route("/predictDog", methods=["POST"])
def predictDog():
    file = request.files['dog_image']

    if file.filename == '':
        # No selected file
        return render_template("index.html", prediction="No file selected")
    
    image = load_img(io.BytesIO(file.read()))
    desired_size = (224, 224)
    image = image.resize(desired_size)

    img_array = tfimage.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class of the image
    predictions = MobileNet.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Top 3 predictions

    # Display the top predictions
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        result = label + " (" + str(round(score*100, 2)) + "%)"

    return render_template("index.html", prediction=result)



if __name__ == '__main__':
    app.run(debug=True)#host='0.0.0.0', debug=True)