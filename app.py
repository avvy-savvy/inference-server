import os
import pandas as pd
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/danger_scale', methods=['GET'])
def get_danger_scale():
    scales = ['Low', 'Moderate', 'Considerable', 'High', 'Extreme']
    return {str(i + 1): label for i, label in enumerate(scales)}


@app.route('/treeline_labels', methods=['GET'])
def get_treeline_labels():
    print(f"App URI: {os.environ.get('HOST_URI')}")
    # TODO add validations on the args and perform inference
    latitude = request.args.get('lat')
    longitude = request.args.get('long')
    datetime = request.args.get('datetime')
    print(f'Requested treeline labels for\n'
          f'  latitude: {latitude}, \n'
          f'  longitude: {longitude}, \n'
          f'  datetime: {datetime}')
    return {
        'Above Treeline': '2',
        'Near Treeline': '2',
        'Below Treeline': '1'
    }


@app.route('/query_arrival_info', methods=['POST'])
def query_arrival_info():
    # global additional_feature
    # global model

    # Load additional features
    additional_feature = pd.read_csv("data/additional_feature.csv", keep_default_na=False)
    # Load model
    model = tf.keras.models.load_model("data/tf_rand_val_nn_v1")

    request_data = request.get_json()

    print(f"App URI: {os.environ.get('HOST_URI')}")
    # TODO add validations on the args and perform inference
    print(f'Requested arriavl info for\n'
          f'  aspect: {request_data["aspect"]}, \n'
          f'  elevation: {request_data["elevation"]}, \n'
          f'  date: {request_data["date"]}')

    input_features = pd.DataFrame(
        {'date_time': [request_data['date']], 'combined_terrain_aspects': [request_data['aspect']],
         'combined_terrain_elevations': [request_data['elevation']]})
    all_features = pd.merge(input_features, additional_feature, on='date_time', how='inner')

    input_dataset = df_to_dataset(all_features)
    predicted = model.predict(input_dataset)
    predictions = np.transpose(predicted)[0]

    return jsonify({
        'prob': str(predictions[0])
    })


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = None  # Initialize labels as None

    # Remove the target column if it exists
    if 'target' in df.columns:
        labels = df.pop('target')  # Remove and store the target column

    # Convert the dataframe to a dictionary of lists
    df_dict = {key: value.tolist() for key, value in df.items()}

    # Create a dataset from the dictionary and labels
    ds = tf.data.Dataset.from_tensor_slices((df_dict, labels))

    # Shuffle the dataset if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    # Batch and prefetch the dataset
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)

    return ds


if __name__ == '__main__':
    app.run(host=os.environ.get('HOST_URI', "0.0.0.0"), port=os.environ.get('HOST_PORT', 5000))
