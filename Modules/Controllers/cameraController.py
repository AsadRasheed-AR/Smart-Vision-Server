from imutils.video import VideoStream,FileVideoStream
import imutils
import time
import threading
import pathlib
from collections import defaultdict
from io import StringIO

import numpy as np
import tensorflow as tf
# import edgeiq

#Libraries for Object Detection
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class camController:
    
    def __init__(self,object_detect,camSource=0,resolution=(320,240),vs=None,showProcessedVideo=False):
        self.vsObj = vs
        self.camSource = camSource
        self.resolution = resolution
        self.outputFrame = None
        self.currentFrame = None
        # self.showProcessedVideo = showProcessedVideo
        self.showProcessedVideo = True
        self.lock = threading.Lock()

        #Model to Detect Objects
        self.object_detect = object_detect
        # self.object_detect.load(engine=edgeiq.Engine.DNN)
    
    def initializeCamera(self):
        _vs = VideoStream(src=self.camSource).start()
        # _vs.start()
        time.sleep(2.0)
        self.vsObj = _vs

    def readFrame(self):
        while True:
            frame = self.vsObj.read()
            with self.lock:
                self.currentFrame = frame.copy()
            
            if (not(self.showProcessedVideo)):
                with self.lock:
                    self.outputFrame = frame.copy()
            # else:
            #     result = self.object_detect.detect_objects(frame,confidence_level=.5)
            #     predictions = edgeiq.filter_predictions_by_label(result.predictions,['person'])
            #     frame = edgeiq.markup_image(frame,predictions, colors=self.object_detect.colors)
            #     self.outputFrame = frame.copy()


    
    def detectObject(self):
        curr_path = str(pathlib.Path.cwd())
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = curr_path
        PATH_TO_LABELS  += '/Object_Detection/Labels/label_map.pbtxt'
        #PATH_TO_LABELS = './Object_Detection/Labels/label_map.pbtxt'
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        # a=2
        while (True):
            frame = None
            with self.lock:
                # frame = self.currentFrame.copy()
                frame = self.vsObj.read()
            output_dict = self.run_inference_for_single_image(self.object_detect,frame)

            #Function to Convert tensor to Image with labels and Bounding Box
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
                    
            if (self.showProcessedVideo):
                with self.lock:
                    self.outputFrame = frame.copy()
    
    def getOutputFrame(self):
        with self.lock:
            frame = self.outputFrame.copy()
        
        return frame
    
    def stop(self):
        with self.lock:
            self.vsObj.stop()

    def startAsyncOperations(self):
        asyncReadFrame = threading.Thread(target=self.readFrame)
        asyncDetectObject = threading.Thread(target=self.detectObject)
        asyncReadFrame.daemon = asyncDetectObject.daemon = True
        asyncReadFrame.start()
        asyncDetectObject.start()
    
    def switchVideo(self,data):
        self.showProcessedVideo = data['showProcessedVideo']
        return ({'showProcessedVideo' : self.showProcessedVideo})
    
    def getVideoStatus(self):
        return ({'showProcessedVideo' : self.showProcessedVideo})


    def run_inference_for_single_image(self,model, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        output_dict = model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
        #Filter Predictions to detect only Persons
        indices = np.argwhere(output_dict['detection_classes'] == 1)
        output_dict['detection_boxes'] = np.squeeze(output_dict['detection_boxes'][indices],axis=1)
        output_dict['detection_scores'] = np.squeeze(output_dict['detection_scores'][indices],axis=1)
        output_dict['detection_classes'] = np.squeeze(output_dict['detection_classes'][indices],axis=1)
        
        return output_dict
