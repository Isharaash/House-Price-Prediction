from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the request
        data = request.form.to_dict()
        
        # Convert input data to list of lists
        input_data = [[float(value) for value in data.values()]]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return prediction as JSON response
        return jsonify({'MedHouseVal': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
