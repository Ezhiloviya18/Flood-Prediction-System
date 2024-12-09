from flask import Flask, request, jsonify, render_template, send_file
import Floodprop as fp  # Import your ML code

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state', methods=['POST'])
def state():
    state_name = request.form['state_name']
    return render_template('predict.html', state=state_name)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        state_name = request.args.get('state_name')
        input1 = float(request.args.get('input1'))
        input2 = float(request.args.get('input2'))
        input3 = float(request.args.get('input3'))

        print(f"Debug: state_name={state_name}, input1={input1}, input2={input2}, input3={input3}")

        if any(val is None or val == '' for val in [input1, input2, input3]):
            return jsonify({'error': 'Input values cannot be empty.'})

    # Convert to float
        try:
            input1 = float(input1)
            input2 = float(input2)
            input3 = float(input3)
        except ValueError as e:
            return jsonify({'error': f'Error converting input values to float: {str(e)}'})



        # Call your machine learning model to get the prediction
        result= fp.fpredict(state_name, input1, input2, input3)
        print("Debug: Result from fp.fpredict:", result)
        try:
            accuracy, graph_path = result
        except (TypeError, ValueError) as e:
            return jsonify({'error': f'Unexpected result from fp.fpredict: {str(e)}'})


        if accuracy.astype(int) == 0:
            prediction = "No Flood Worries"
        else:
            prediction = "High risk of flood"

        response = {
            'prediction': prediction,
            'graphPath': graph_path
        }

        return jsonify(response)


    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True, port=8080)
