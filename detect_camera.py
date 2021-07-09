##
# @file: detect_camera.py
# @author: SIANA Systems
# @date: 07/21
# @ref: https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py
#
# This is the original Google demo adpated for the MPCam camera. Instead of
# taking an image, this script uses the camera feed and outputs the result
# through a web streamer.
#
#--->> Google License <<-------------------------------------------------------
#
# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral on MPCam to detect objects using the camera.

Example usage:
```
python3 detect_camera.py
```
"""

import platform
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import tflite_runtime
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, get_runtime_version

from pympcam.coralManager import CoralManager
from imutils.video import VideoStream, ImageOutput, FPS

#--->> TUNABLES <<-------------------------------------------------------------

verbose = False

threshold = 0.4

enable_web_output = True

#--->> DEFAULTS <<-------------------------------------------------------------

model_file = "model/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
labels_file = "coco_labels.txt"

camera_width = 640
camera_height = 480
camera_fps = 15

#------------------------------------------------------------------------------

def draw_metrics(image, fps, model_msec=0, numb_objs=0):
    '''Draws the FPS & model duration on top of the input image.

       Args:
          image: a cv2 image
          fps: the Frame-per-second value   
          model_msec: the model compute duration in msec
          numb_objs: the # of objects

      Returns:
          CV2 image = image+footer
    '''
    # create white(255) footer image (width x 20)...
    footer = np.zeros((20, camera_width, 3), np.uint8) + 255

    # draw metrics on footer...
    footer = cv2.putText(footer, 
                        'FPS: {:.2f}, TPU: {:.0f} ms, Objs: {}'.format(
                            float(round(fps,2)), 
                            float(round(model_msec)),
                            numb_objs),
                        (15, 15),                 # bottom-left corner x/y 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6,                      # fontScale factor
                        (0,0,0),                  # color (r,g,b)
                        1,                        # thickness (px)
                        cv2.LINE_AA
                        )
                        
    # return appended footer to original image...
    return cv2.vconcat([image, footer])

def draw_objects(src, objs, labels):
  '''Draws the bounding box and label for each object.

     Args:
        src: CV2 source image
        objs: array of objects for which to draw bounding boxes
        labels: dic(id,string) of labels for the obj.id

     Returns:
        CV2 processed image   
  '''

  color = (36,255,12) # BGR!

  for obj in objs:
    bbox = obj.bbox

    src = cv2.rectangle(src,
                        (bbox.xmin, bbox.ymin),   # top-left
                        (bbox.xmax, bbox.ymax),   # bottom-right
                        color,                    # BGR
                        3                         # line thickness
                        )

    src = cv2.putText(src,
                      '%s: %.2f' % (labels.get(obj.id, obj.id), obj.score),
                      (bbox.xmin + 10, bbox.ymin -5),   # bottom-left x/y
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7,    # font, fontScale
                      color,                            # BGR 
                      2                                 # thickness
                      )
  return src

def process_frame(interpreter, input_image, threshold):
  '''Processes a new frame with the given model.

      Args:
        interpreter: the TFLite interpreter
        input_image: the input RGB image

      Returns:
         tuple(objs, duration)
         . image: array of detected objects or None. 
         . duration: inference duration in msec.
  '''
  #start = time.perf_counter() 

  start = time.perf_counter()

  # resizing to fit model...
  if verbose: print("!! resizing & loading into interpreter...")
  _, scale = common.set_resized_input(
            interpreter, 
            input_image.size, 
            lambda size: input_image.resize(size, Image.NEAREST)
            ) 
    
  # run inference...
  if verbose: print("!! running inference...")  
  interpreter.invoke()

  # process detected objects...  
  objs = detect.get_objects(interpreter, threshold, scale) 
  if verbose: print("!! OBJECTS:\n{}".format(objs))

  inference_time = time.perf_counter() - start

  return objs, inference_time

#------------------------------------------------------------------------------

def main():
  global model_file, labels_file, threshold
  global camera_width, camera_height, camera_fps

  print("\n** MPCam: PyCoral Camera Detector **\n")
  print(">> system info:")
  print("\tpython {}".format(platform.python_version()))
  print("\topencv {}".format(cv2.__version__))
  print("\ttflite {}".format(tflite_runtime.__version__))
  print("\tedgtpu {}".format( get_runtime_version() )) 

  print(">> turning on the coral...")
  coral = CoralManager()
  coral.turnOn()

  print(">> configuring camera...")    
  camera = VideoStream()
  cv = camera.stream
  print("\tWidth/Height: {}/{}".format(camera_width, camera_height))
  cv.stream.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
  cv.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
  print("\tFPS: {}".format(camera_fps))
  cv.stream.set(cv2.CAP_PROP_FPS, camera_fps)
  camera.start()

  if enable_web_output:
    print(">> configuring web streamer: http://mpcam.local:8080"),
    display = ImageOutput(screen=False)
  else:
    print(">> web streamer is disabled!")

  try:
    print(">> loading the model: {}".format(model_file))
    labels = read_label_file(labels_file)
    interpreter = make_interpreter(model_file)
    interpreter.allocate_tensors()
    print(">> model input size: {}".format(common.input_size(interpreter)))

    print(">> start pocessing...")
    last_print = datetime.now()
    # fps/time trackers
    global_fps = FPS().start()
    inference_duration = nframes = 0
    # reported metrics
    report_inference = report_fps = 0
    while True:

      # grab frame
      cv2_frame = camera.read()
      global_fps.update()
      nframes += 1

      # convert cv2 BGR frame into RGB image
      input_image = Image.fromarray(cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB))

      # run detector on input image...
      detected_objs, duration = process_frame(interpreter, input_image, threshold)
      inference_duration += duration   

      numb_objects = 0 if detected_objs is None else len(detected_objs)   

      if enable_web_output:        
        # convert RGB image to cv2 BGR
        cv2_frame = cv2.cvtColor(np.asarray(input_image),cv2.COLOR_RGB2BGR) 
        # draw bounding boxes
        cv2_frame = draw_objects(cv2_frame, detected_objs, labels)
        # overlay metrics                
        cv2_frame = draw_metrics(cv2_frame, report_fps, report_inference, numb_objects)
        # output processed frame
        display.stream('detector', cv2_frame)
            
      # print metrics in console every 5sec
      now = datetime.now()
      if (now - last_print).total_seconds() > 5:
        last_print = now
        print(">> FPS: {:.2f}, TPU: {:.0f} ms, Objs: {}".format(report_fps, report_inference, numb_objects))                  

      # time to update metrics?
      if nframes > 10:
        # update global fps
        global_fps.stop()
        report_fps = global_fps.fps()
        # update averaged inference duration
        report_inference = (inference_duration / nframes) * 1000
        inference_duration = nframes = 0

  except BrokenPipeError:
    print(">> stream aborted: restart...")
  
  except KeyboardInterrupt:
    print(">> user abort...")

  finally:
    print(">> turn off camera & coral...")
    camera.stop()
    coral.turnOff()
    print(">> done!")

if __name__ == '__main__':
  main()
