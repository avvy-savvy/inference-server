import json
import os
from math import sin, cos, atan2, sqrt, radians

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/danger_scale', methods=['GET'])
@cross_origin()
def get_danger_scale():
    scales = ['Low', 'Moderate', 'Considerable', 'High', 'Extreme']
    return {str(i + 1): label for i, label in enumerate(scales)}


@app.route('/treeline_labels', methods=['GET'])
@cross_origin()
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


@app.route('/query_trip_possibility', methods=['POST'])
@cross_origin()
def query_trip_possibility():
    # global additional_feature
    # global model

    # Load additional features
    weather_lookup = pd.read_csv("data/model-1/weather_data_m1.csv", keep_default_na=False)
    # Load model
    model = joblib.load("data/model-1/avvy_predict_weather.joblib")

    request_data = request.get_json()
    lat = float(request_data['lat'])
    lng = float(request_data['lng'])

    # TODO add validations on the args and perform inference
    print(f'Requested arriavl info for\n'
          f'  latitude: {request_data["lat"]}, \n'
          f'  longitude: {request_data["lng"]}, \n'
          f'  date: {request_data["date"]}')

    user_input = weather_lookup[weather_lookup['date_time'] == request_data["date"]]
    print(len(user_input))

    distances = [haversine(lat, lng, row['lat'], row['lng']) for _, row in user_input.iterrows()]
    nearest_index = np.argmin(distances)
    user_input = user_input.iloc[[nearest_index]].drop(columns=['date_time', 'is_avy_obs', 'Unnamed: 0'])

    predicted = model.predict(user_input)
    predictions = np.transpose(predicted)

    return jsonify({
        'prob': str(predictions[0])
    })


@app.route('/query_arrival_info', methods=['POST'])
@cross_origin()
def query_arrival_info():
    # global additional_feature
    # global model

    # Load additional features
    additional_feature = pd.read_csv("data/model-2/additional_features.csv", keep_default_na=False)
    # Load model
    model = tf.keras.models.load_model("data/model-2/tf_rand_val_nn_v3")

    request_data = request.get_json()

    # TODO add validations on the args and perform inference
    print(f'Requested arriavl info for\n'
          f'  aspect: {request_data["aspect"]}, \n'
          f'  elevation: {request_data["elevation"]}, \n'
          f'  date: {request_data["date"]}')

    input_features = pd.DataFrame(
        {'date_time': [request_data['date']], 'combined_terrain_aspects': [request_data['aspect']],
         'combined_terrain_elevations': [request_data['elevation']]})
    all_features = pd.merge(input_features, additional_feature, on='date_time', how='left')

    input_dataset = df_to_dataset(all_features, shuffle=False)
    predicted = model.predict(input_dataset)
    predictions = np.transpose(predicted)[0]

    return jsonify({
        'prob': str(predictions[0])
    })


@app.route('/query_spot_details', methods=['POST'])
@cross_origin()
def query_spot_details():
    # global additional_feature
    # global model

    request_data = request.get_json()
    lat = float(request_data['lat'])
    lng = float(request_data['lng'])

    # Load unique stids
    unique_stid_lat_lon = pd.read_csv("data/model-3/unique_stid_lat_lon.csv", keep_default_na=False)
    unique_stid_lat_lon['Latitude'] = pd.to_numeric(unique_stid_lat_lon['Latitude'], errors='coerce')
    unique_stid_lat_lon['Longitude'] = pd.to_numeric(unique_stid_lat_lon['Longitude'], errors='coerce')
    unique_stid_lat_lon = unique_stid_lat_lon.dropna(subset=['Latitude', 'Longitude'])
    unique_stid_lat_lon['lat_lng'] = list(zip(unique_stid_lat_lon['Latitude'], unique_stid_lat_lon['Longitude']))

    # Get nearest station
    distances = [haversine(lat, lng, row['Longitude'], row['Longitude']) for _, row in unique_stid_lat_lon.iterrows()]
    nearest_index = np.argmin(distances)
    station_id = unique_stid_lat_lon.iloc[nearest_index]['STID']

    # create user input df
    user_input = pd.DataFrame(
        {'date_time': [request_data['date']], 'combined_terrain_aspects': [request_data['aspect']],
         'combined_terrain_elevations': [request_data['elevation']], 'lat': [lat], 'lng': [lng],
         'lat_lng': [(lat, lng)], 'STID': [station_id]})

    # Find weather data
    pivot_weather = pd.read_csv("data/model-3/pivot_weather.csv", keep_default_na=False)

    # merge user-input with weather
    removed = ['STID', 'Latitude', 'Longitude', 'lat', 'lng', 'lat_lng']
    merged_df = pd.merge(user_input, pivot_weather, on=['STID', 'date_time'], how='left').drop(columns=removed)

    # Load additional features
    additional_features = pd.read_csv("data/model-3/additional_features.csv", keep_default_na=False)

    # Load model
    model = tf.keras.models.load_model("data/model-3/model_3_v1")

    # TODO add validations on the args and perform inference
    print(f'Requested spot details for\n'
          f'  aspect: {request_data["aspect"]}, \n'
          f'  elevation: {request_data["elevation"]}, \n'
          f'  date: {request_data["date"]}, \n'
          f'  latitude: {request_data["lat"]}, \n'
          f'  longitude: {request_data["lng"]}')

    all_features = pd.merge(merged_df, additional_features,
                            on=['date_time', 'combined_terrain_aspects', 'combined_terrain_elevations'], how='left',
                            suffixes=('', '_drop'))
    all_features = all_features.filter(regex='^(?!.*_drop)')

    with open(f"data/model-3/feature_dict.yaml", "r") as f:
        feature_dict = json.load(f)

    all_features[feature_dict["num_vars_norm"]] = (
        all_features[feature_dict["num_vars_norm"]].fillna(-1).replace("", -1).astype(float).astype(int)
    )
    all_features['target'] = all_features['is_avy_obs'].replace({'Yes': 1, 'No': 0})

    input_dataset = df_to_dataset_wo_dups(all_features, feature_dict, shuffle=False)
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


def df_to_dataset_wo_dups(dataframe, feature_dict, buffer_size=None, shuffle=True, batch_size=32):
    labels = dataframe["target"]
    df = {}
    for key, cols in feature_dict.items():
        if key in ["num_vars_norm", "num_vars"]:
            df[key] = dataframe[cols]
        else:
            for col in cols:
                df[col] = dataframe[col].tolist()
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius_earth = 3959  # Radius of Earth in kilometers (6371) or for miles (3959)

    # Calculate the distance
    distance = radius_earth * c

    return distance


if __name__ == '__main__':
    app.run(host=os.environ.get('HOST_URI', "0.0.0.0"), port=os.environ.get('HOST_PORT', 5000))
