from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "hello world"


@app.route('/predict', methods=['post'])
def predict():
    nitrogen = request.form.get('nitrogen')
    phosphorus = request.form.get('phosphorus')
    potassium = request.form.get('potassium')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')
    input_query = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall])

    result = model.predict(input_query.reshape(1, -1))

    return jsonify({"result": str(result[0])})


if __name__ == '__main__':
    app.run(debug=True)
