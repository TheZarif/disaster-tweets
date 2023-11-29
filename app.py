from flask import Flask, render_template, request      
import joblib

app = Flask(__name__)

class Model:
    def __init__(self, model_type, model_obj):
        self.model_type = model_type
        self.model_obj = model_obj

    def predict(self, text):
        if self.model_type == 'svm':
            res = self.model_obj.predict([text])
            return res[0] == 1

def load_model_from_disk(model):
    if model == 'svm':
        loaded_pipeline = joblib.load('best_svm_pipeline.pkl')
        return Model('svm', loaded_pipeline)

# Load your model and store it as a global variable
model = load_model_from_disk('svm')

@app.route('/', methods=['GET', 'POST'])
def index():
    tweet = None
    message = None

    if request.method == 'POST':
        tweet = request.form.get('tweet')
        res = model.predict(tweet)
        message = f'Disaster: {res}'

    return render_template('home.html', tweet=tweet, message=message)

@app.route('/hello_world')
def hello_world():
    return 'Hello, Flask!'