##
# @file: detect_image.py
# @author: SIANA Systems
# @date: 07/21
# @ref: https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py
#
# This is the original Google demo adpated for the MPCam camera. This script 
# processes an image and outputs the result through a web streamer.
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
r"""Example using PyCoral on the MPCam to detect objects in a given image.

Example usage:
```
python3 detect_image.py
```
"""

import platform
import time

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw

import tflite_runtime
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, get_runtime_version

from pympcam.coralManager import CoralManager
from imutils.video import ImageOutput

#--->> DEFAULTS <<-------------------------------------------------------------

model = "model/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
labels = "coco_labels.txt"
threshold = 0.4
count = 10

#image_file = "grace_hopper.bmp"
image_file = "people.jpg"

#------------------------------------------------------------------------------

def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='green', width=3)
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='green')

#------------------------------------------------------------------------------

def main():
  global model, labels, image_file, threshold, count

  print("\n** MPCam: PyCoral Image Detector **\n")
  print("\tpython {}".format(platform.python_version()))
  print("\topencv {}".format(cv2.__version__))
  print("\ttflite {}".format(tflite_runtime.__version__))
  print("\tedgtpu {}".format( get_runtime_version() )) 

  print("> turning on the coral...")
  coral = CoralManager()
  coral.turnOn()
  time.sleep(2)

  print("> enabling the display...")
  display = ImageOutput(screen=False)

  print("> loading the model: {}".format(model))
  labels = read_label_file(labels)
  interpreter = make_interpreter(model)
  interpreter.allocate_tensors()

  image = Image.open(image_file)
  print("> input image size: {}".format(image.size))
  _, scale = common.set_resized_input(
                  interpreter, 
                  image.size, 
                  lambda size: image.resize(size, Image.ANTIALIAS)
                  )

  print('----INFERENCE TIME----')
  print('Note: The first inference is slow because it includes',
        'loading the model into Edge TPU memory.')
  for _ in range(count):
    start = time.perf_counter()
    interpreter.invoke()    
    objs = detect.get_objects(interpreter, threshold, scale)
    inference_time = time.perf_counter() - start
    print('%.2f ms' % (inference_time * 1000))

  print('-------RESULTS--------')
  if not objs:
    print('No objects detected')

  for obj in objs:
    print(labels.get(obj.id, obj.id))
    print('  id:    ', obj.id)
    print('  score: ', obj.score)
    print('  bbox:  ', obj.bbox)

  image = image.convert('RGB') 
  draw_objects(ImageDraw.Draw(image), objs, labels)

  # convert image to CV:UMat
  cv_image = numpy.array(image)
  cv_image = cv_image[:, :, ::-1].copy()

  print("> updated display on http://mpcam.local:8080/test")
  display.show('test', cv_image)
  display.waitForKey()

if __name__ == '__main__':
  main()
