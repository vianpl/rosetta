If you’re using a file video stream, then comment Line vs = VideoStream(src=0).start() and fileStream = False.

Otherwise, if you want to use a built-in webcam or USB camera, uncomment Line vs = VideoStream(src=0).start() 
and fileStream = False

to run the detection of video
python detect_blinks.py \
	--shape-predictor shape_predictor_68_face_landmarks.dat \
	--video blink_detection_demo.mp4

to run the detection over webcam
2python detect_blinks.py \
	--shape-predictor shape_predictor_68_face_landmarks.dat
