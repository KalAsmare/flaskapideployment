from flask import Flask, request, jsonify
import numpy as np
import pickle

pickled_model = pickle.load(open('model.pickle', 'rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():

    age= request.form.get('age')
    marital_status = request.form.get('marital_status')
    educational_status = request.form.get('educational_status')
    number_of_birth = request.form.get('number_of_birth')
    history_of_sti = request.form.get('history_of_sti ')
    hiv_test_result = request.form.get('hiv_test_result')
    target_population_category = request.form.get('target_population_category')
    hiv_positive_linked_wit_art = request.form.get('hiv_positive_linked_wit_art')

    input_query = np.array([[age, marital_status, educational_status, number_of_birth, history_of_sti, hiv_test_result, target_population_category, hiv_positive_linked_wit_art]])
    result = pickled_model.predict(input_query)[0]
    return jsonify({'cervical':str(result)})

#this commands the script to run in the given port
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)