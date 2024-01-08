# Model-deployment
Model deployment example using Flask and [Render](https://render.com/)

![preview](https://github.com/avrambardas/Model-deployment/blob/Spam-detection/images/preview.png)

Live preview [here](https://email-spam-detection-preview.onrender.com/)

You can run the code on github codespace with the following steps:

1. create codna env: conda create -n model_deployment python=3.10.0
2. activate env: conda activate model_deployment
3. install the following: pip install flask gunicorn pickle-mixin scikit-learn
4. run app: python app.py
5. deactivate env: conda deactivate
