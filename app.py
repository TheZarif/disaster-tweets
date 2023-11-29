from flask import Flask, render_template, request      

app = Flask(__name__)

class Model:
    def __init__(self, on):
        if on:
            self.on = True

    def predict(self, text):
        return self.on

def load_model_from_disk():
    return Model(True)

# Load your model and store it as a global variable
model = load_model_from_disk()

@app.route('/', methods=['GET', 'POST'])
def index():
    tweet = None
    message = None

    if request.method == 'POST':
        tweet = request.form.get('tweet')
        res = model.predict(tweet)
        message = f'You submitted the following tweet: "{tweet}" Disaster: {res}'

    return render_template('home.html', tweet=tweet, message=message)

@app.route('/hello_world')
def hello_world():
    return 'Hello, Flask!'