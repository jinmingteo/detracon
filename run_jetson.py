#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import cv2
import jetson.inference
import jetson.utils

import argparse
import sys
import time

from deep_sort_realtime.deepsort_tracker_emb import DeepSort as Tracker
from utils.drawer import Drawer

f = open('coco.txt','r')
labels = f.read()
label_arr = labels.split('\n')

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
								 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
																					   jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# assuming 7fps & 70nn_budget, tracker looks into 10secs in the past.
nn_budget = 70
tracker = Tracker(
	max_age=30, nn_budget=nn_budget, override_track_class=None)
drawer = Drawer()
# process frames until the user exits
while True:
	tic = time.time()
	# capture the next image
	img = input.Capture()
	np_source = jetson.utils.cudaToNumpy(img)
	np_source = cv2.cvtColor(np_source, cv2.COLOR_RGBA2BGR)
	# detect objects in the image (with overlay)
	#detections = net.Detect(img, overlay=opt.overlay)
	detections = net.Detect(img, overlay='none')
	chosen_track = None

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))
	raw_dets = []
	for detection in detections:
		bbox = [detection.Left, detection.Top, detection.Width, detection.Height]
		class_name = label_arr[detection.ClassID]
		raw_dets.append((bbox, detection.Confidence, class_name))
		print(detection)

	tracks = tracker.update_tracks(np_source, raw_dets)

	show_frame = np_source.copy()
	if raw_dets: 	
		drawer.draw_tracks(
			show_frame,
			tracks,
			chosen_track=chosen_track
		)
	#show_frame = cv2.cvtColor(show_frame, cv2.COLOR_BGR2RGBA)
	#display_img = jetson.utils.cudaFromNumpy(show_frame)
	toc = time.time()
	fps = 1/(toc-tic)
	cv2.putText(show_frame, f'fps: {fps:.3f}', (200,25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
	cv2.imshow("webcam", show_frame)
	k = cv2.waitKey(16)
	if k == ord('q'):
		break
	# render the image
	#output.Render(display_img)

	# update the title bar
	#output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
