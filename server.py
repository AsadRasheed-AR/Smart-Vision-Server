# import the necessary packages

#Controller Modules
from Modules.Controllers.cameraController import camController
from Modules.Controllers.requestController import reqController

#Libraries for webServer
from flask import Response
from flask import Flask,request, jsonify
from flask import render_template

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




# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# fps = edgeiq.FPS().start()


#Object Detection Initialization
# obj_detect = edgeiq.ObjectDetection(
#         "alwaysai/mobilenet_ssd")
# obj_detect.load(engine=edgeiq.Engine.DNN)

# print("Loaded model:\n{}\n".format(obj_detect.model_id))
# print("Engine: {}".format(obj_detect.engine))
# print("Accelerator: {}\n".format(obj_detect.accelerator))
# print("Labels:\n{}\n".format(obj_detect.labels))


#Load Model
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile	

#Path of Current Working Directory
curr_path = str(pathlib.Path.cwd())
# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = curr_path
# PATH_TO_LABELS  += '/Object_Detection/Labels/label_map.pbtxt'
# #PATH_TO_LABELS = './Object_Detection/Labels/label_map.pbtxt'
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#Path Of Model
model_path = curr_path
model_path  += '/Object_Detection/Model/ssd_mobilenet_v1_coco_2017_11_17'
#Load Model
model = tf.saved_model.load(model_path)
model = model.signatures['serving_default']

#initialize Controllers
cc = camController(object_detect=model)
cc.initializeCamera()
cc.startAsyncOperations()

rc = reqController()




# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
# vs = VideoStream(src=0,resolution=(640,480)).start()
# vs = FileVideoStream('demo1.mp4').start()
vs = None
# time.sleep(2.0)


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

# def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	# global vs, outputFrame, lock
	# initialize the motion detector and the total number of frames
	# read thus far
	# md = SingleMotionDetector(accumWeight=0.1)
	# total = 0

    	# loop over frames from the video stream
	# while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		# frame = vs.read()
        # results = obj_detect.detect_objects(frame, confidence_level=.5)
        # frame = edgeiq.markup_image(frame, results.predictions, colors=obj_detect.colors)
		#frame = imutils.resize(frame, width=400)
		# results = obj_detect.detect_objects(frame, confidence_level=.5)
		# predictions = edgeiq.filter_predictions_by_label(results.predictions,['person'])
		# frame = edgeiq.markup_image(
                        # frame, predictions, colors=obj_detect.colors)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# gray = cv2.GaussianBlur(gray, (7, 7), 0)
		# # grab the current timestamp and draw it on the frame
		# timestamp = datetime.datetime.now()
		# cv2.putText(frame, timestamp.strftime(
		# 	"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # results = obj_detect.detect_objects(frame,confidence_level=.5)
        # frame = edgeiq.markup_image(frame,results.predictions,colors=obj_detect.colors)

        # Generate text to display on streamer
        # for prediction in results.predictions:
        #     text.append("{}: {:2.2f}%".format(prediction.label, prediction.confidence * 100))
        

        # if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
		# if total > frameCount:
		# 	# detect motion in the image
		# 	motion = md.detect(gray)
		# 	# check to see if motion was found in the frame
		# 	if motion is not None:
		# 		# unpack the tuple and draw the box surrounding the
		# 		# "motion area" on the output frame
		# 		(thresh, (minX, minY, maxX, maxY)) = motion
		# 		cv2.rectangle(frame, (minX, minY), (maxX, maxY),
		# 			(0, 0, 255), 2)
		
		# # update the background model and increment the total number
		# # of frames read thus far
		# md.update(gray)
		# total += 1
		# acquire the lock, set the output frame, and release the
		# lock
		# frame = cc.getOutputFrame()
		# with lock:
			# outputFrame = frame.copy()


def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		# with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			# if outputFrame is None:
			# 	continue
			# encode the frame in JPEG format
			# (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			# if not flag:
			# 	continue
		# yield the output frame in the byte format
		frame = cc.getOutputFrame()
		if frame is None:
			continue
		(flag,encodedImage) = cv2.imencode(".jpg",frame)
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
		# fps.update()
		# yield(outputFrame)


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

	

	# start a thread that will perform motion detection
	# t = threading.Thread(target=detect_motion, args=(
	# 	args["frame_count"],))
	# t.daemon = True
	# t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
# release the video stream pointer
# vs.stop()
# fps.stop()
print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
print("approx. FPS: {:.2f}".format(fps.compute_fps()))
cc.stop()

