from flask import Flask, request

app = Flask(__name__)


@app.route('/danger_scale', methods=['GET'])
def get_danger_scale():
    scales = ['Low', 'Moderate', 'Considerable', 'High', 'Extreme']
    return {str(i + 1): label for i, label in enumerate(scales)}


@app.route('/treeline_labels', methods=['GET'])
def get_treeline_labels():
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
