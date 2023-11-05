from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)  # Initialize CORS extension

# Define the directory where CSV files are located
csv_directory = 'build'

@app.route('/fx/data/<pair>', methods=['GET'])
def get_fx_data(pair):
    try:
        # Read the CSV file
        csv_file_path = f"{csv_directory}/{pair}-live-processed.csv"
        data = pd.read_csv(csv_file_path)

        # Convert the CSV data to a list of dictionaries
        data_list = data.to_dict(orient='records')

        # Create a response dictionary
        response = {
            'data': data_list[:50]
        }

        return jsonify(response), 200
    except FileNotFoundError:
        return jsonify({'error': 'Pair not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)