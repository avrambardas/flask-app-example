from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
cv = pickle.load(open("models/EmailClassifier/cv.pkl", "rb"))
clf = pickle.load(open("models/EmailClassifier/clf.pkl", "rb"))

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


if __name__ == '__main__':
    app.run(debug=True)#host='0.0.0.0', debug=True)