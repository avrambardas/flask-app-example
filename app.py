from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import Image
import pickle
import io

app = Flask(__name__)
cv = pickle.load(open("models/EmailClassifier/cv.pkl", "rb"))
clf = pickle.load(open("models/EmailClassifier/clf.pkl", "rb"))
#model_ResNet50 = load_model('models/DogClassifier/resnet50_model.h5')
model_ResNet50 = ResNet50(weights='imagenet')

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
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model_ResNet50.predict(image)
    label = decode_predictions(yhat)
    label = '%s (%.2f%%)' % (label[0][0][1], label[0][0][2]*100)
    return render_template("index.html", prediction=label)



if __name__ == '__main__':
    app.run(debug=True)#host='0.0.0.0', debug=True)