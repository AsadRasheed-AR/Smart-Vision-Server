# import the necessary packages

#Controller Modules
from Modules.Controllers.cameraController import camController
from Modules.Controllers.requestController import reqController
from Modules.Controllers.nodemcuController import espController

#Libraries for webServer
from flask import Response
from flask import Flask,request, jsonify
from flask import render_template
from flask_socketio import *

#Libraries for Object Detection
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#Libraries for Processing and Object Detection
import numpy as np
import tensorflow as tf
import cv2
from imutils.video import VideoStream,FileVideoStream
import threading
import imutils

#Libraries for Local Visualization and debugging
from matplotlib import pyplot as plt
#from PIL import Images

#Other Libraries for utilities
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
import argparse
import datetime
import time

# import edgeiq

# initialize a flask object
app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

socketio = SocketIO(app, cors_allowed_origins="*")


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

#Load Model
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile	

#Path of Current Working Directory
curr_path = str(pathlib.Path.cwd())

#Path Of Model
model_path = curr_path
model_path  += '/Object_Detection/Model/ssd_mobilenet_v1_coco_2017_11_17'
#Load Model
model = tf.saved_model.load(model_path)
model = model.signatures['serving_default']

#initialize Controllers
rc = reqController()

nc = espController(rc_obj=rc)
nc.startAsyncOperations()

cc = camController(object_detect=model,rc_obj=rc)
cc.initializeCamera()
cc.startAsyncOperations()

vs = None

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

@app.route("/getCurrentStatus", methods=['GET'])
def server_getCurrentStatus():
	# return the rendered template
	return (jsonify(rc.getCurrentStatus()))

@app.route("/getControlStatus", methods=['GET'])
def server_getControlStatus():
	# return the rendered template
	return (jsonify(rc.getControlStatus()))

@app.route("/setCurrentStatus", methods=['POST'])
def server_setCurrentStatus():
	print('setCurrentStatus Request Received')
	content = request.json
	status = rc.setCurrentStatus(content)
	return jsonify(status)


@app.route("/switch_autoControl", methods=['POST'])
def server_switch_autoControl():
	print('switch_autoControl Request Received')
	content = request.json
	print(content)
	status = rc.setControlStatus(content)
	return jsonify(status)


@app.route("/switchVideo" , methods=['POST'])
def server_switchVideo():
	print('switchVideo Request Received')
	content = request.json
	status = cc.switchVideo(content)
	return(status)


@app.route("/getVideoStatus" , methods=['GET'])
def server_getVideoStatus():
	print('getVideoStatus request recived')
	status = cc.getVideoStatus()
	return(status)

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# yield the output frame in the byte format
		frame = cc.getOutputFrame()
		if frame is None:
			continue
		(flag,encodedImage) = cv2.imencode(".jpg",frame)
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	
	socketio.run(app,host=args["ip"], port=args["port"], debug=True,
		 use_reloader=False)

print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
print("approx. FPS: {:.2f}".format(fps.compute_fps()))
cc.stop()

