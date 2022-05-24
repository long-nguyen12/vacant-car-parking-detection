import datetime
import hashlib

import cv2
import tensorflow as tf
from flask import Flask, render_template, Response, request, session, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from pymongo import MongoClient
from tensorflow.python.saved_model import tag_constants

from commons import commonFunctions
from detect import detect

app = Flask(__name__)

jwt = JWTManager(app)
app.config['JWT_SECRET_KEY'] = 'HDU'
app.config['SECRET_KEY'] = 'HDU'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(days=1)

client = MongoClient("mongodb://localhost:27017/")
db = client["parking"]
users_collection = db["users"]
cameras_collection = db["cameras"]

camera = cv2.VideoCapture(
    'rtsp://admin:Admin12345@tronghau8.kbvision.tv:37779/cam/realmonitor?channel=1&subtype=0')  # use 0 for web camera


# camera = cv2.VideoCapture(0)


def gen_frames():
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    ground_truth = commonFunctions.get_ground_truth('./data/ground_truth/video_1.p')
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            image, count = detect(saved_model_loaded, frame, ground_truth)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route("/api/v1/users", methods=["POST"])
def register():
    try:
        new_user = request.get_json()
        new_user["password"] = hashlib.sha256(new_user["password"].encode("utf-8")).hexdigest()
        doc = users_collection.find_one({"username": new_user["username"]})
        if not doc:
            users_collection.insert_one(new_user)
            return jsonify({'msg': 'User created successfully'}), 201
        else:
            return jsonify({'msg': 'Username already exists'}), 409
    except:
        return jsonify({'msg': 'Username already exists'}), 409


@app.route("/api/v1/login", methods=["POST"])
def login():
    login_details = request.get_json()
    user_from_db = users_collection.find_one({'username': login_details['username']})

    if user_from_db:
        encrypted_password = hashlib.sha256(login_details['password'].encode("utf-8")).hexdigest()
        if encrypted_password == user_from_db['password']:
            access_token = create_access_token(identity=user_from_db['username'])
            return jsonify(access_token=access_token), 200

    return jsonify({'msg': 'The username or password is incorrect'}), 401


@app.route("/api/v1/user", methods=["GET"])
@jwt_required()
def profile():
    current_user = get_jwt_identity()
    user_from_db = users_collection.find_one({'username': current_user})
    if user_from_db:
        # del user_from_db['_id'], user_from_db['password']
        return jsonify({'profile': user_from_db}), 200

    return jsonify({'msg': 'Profile not found'}), 404


@app.route('/api/v1/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/v1/camera', methods=["GET"])
@jwt_required()
def get_cameras():
    cameras_from_db = cameras_collection.find()
    if cameras_from_db:
        return jsonify({'data': cameras_from_db}), 200

    return jsonify({'msg': 'Cameras not found'}), 404


@app.route('/api/v1/camera', methods=["POST"])
@jwt_required()
def post_camera():
    new_camera = request.get_json()
    if new_camera:
        cameras_collection.insert_one(new_camera)
        return jsonify({'msg': 'Camera created successfully'}), 201

    return jsonify({'msg': 'Cameras not found'}), 404


@app.route('/api/v1/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
