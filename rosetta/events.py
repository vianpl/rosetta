# -*- coding: utf-8 -*-
# Blink
from scipy.spatial import distance as dist
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from collections import deque
from imutils import face_utils
from pipeline import *
from enum import Enum
import pyaudio
# TODO: get rid of dependency
import imutils
import signal
import dlib
import wave
import sys
import cv2

EYE_AR_THRESH = 0.3
FPS = 30

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(LSTART, LEND) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(RSTART, REND) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

class InputEvents(Enum):
    SKIP = 1
    ENTER = 2
    CLOSE = 3

class Camera(Element):
    def __init__(self, name):
        Element.__init__(self, name, ElementType.SOURCE)

        # start the video stream thread
        self.vs = cv2.VideoCapture(1)
        self.camera_fps = self.vs.get(cv2.CAP_PROP_FPS)
        self.count = 0
        self.drop_rate = round(self.camera_fps / FPS)

    def consume(self, data):
        out = None

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        ret, frame = self.vs.read()
        if ret != True:
            print("frame grab error {}:".format(ret))

        if ret and not (self.count % self.drop_rate):
            # TODO: Find a way to make this more clean
            out = {None: frame}

        self.count += 1

        return out


class BlinkDetector(Element):
    def __init__(self, name):
        Element.__init__(self, name, ElementType.PROCESSING)

    def prepare(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def single_eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        return ear

    def eye_aspect_ratio(self, shape):
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEAR = self.single_eye_aspect_ratio(shape[LSTART:LEND])
        rightEAR = self.single_eye_aspect_ratio(shape[RSTART:REND])

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        return ear

    def consume(self, data):
        frame = imutils.resize(data, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = self.detector(gray, 0)

        if not len(rects) == 1:
            return None

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = self.predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        ear = self.eye_aspect_ratio(shape)

        out_data = {"frame": frame, "shape": shape, "ear": ear}
        out = {None: out_data}
        return out

    def cleanup(self):
        self.vs.stop()

class CameraPlot(Element):
    def __init__(self, name):
        Element.__init__(self, name, ElementType.SINK)

    def prepare(self):
        cv2.startWindowThread()
        cv2.namedWindow("Frame")

    def consume(self, data):
        frame = data["frame"]
        shape = data["shape"]
        ear = data["ear"]

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(shape[LSTART:LEND])
        rightEyeHull = cv2.convexHull(shape[RSTART:REND])
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)

    def cleanup(self):
        cv2.destroyAllWindows()
        self.vs.stop()

class BlinkPlot(Element):
    def __init__(self, name):
        Element.__init__(self, name, ElementType.SINK)

    def prepare(self):
        # Create 10s window buffer. It moves along with the incoming
        # data
        self.cbuf = deque(maxlen=int(10 * FPS))
        for i in range(self.cbuf.maxlen):
            self.cbuf.append(EYE_AR_THRESH)

        self.count = 0

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set_ylim(auto=True)
        self.fig.show()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.hline = self.ax.axhline(EYE_AR_THRESH, color='r')
        self.line, = self.ax.plot(self.cbuf, animated=True)

    def consume(self, data):
        ear = data["ear"]
        self.cbuf.append(ear)

        self.fig.canvas.restore_region(self.background)
        self.line.set_ydata(self.cbuf)

        self.count += 1
        if self.count % 100 == 0:
            self.ax.relim()
            self.ax.autoscale_view()

        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.hline)
        self.fig.canvas.blit(self.ax.bbox)

    def cleanup():
        plt.close()
        self.p.join()

class InputEventFilter(Element):
    def __init__(self, name):
        Element.__init__(self, name, ElementType.PROCESSING)

    def prepare(self):
        self.BLINK_TICK_MS = 400
        self.num_planes_timeout = round(self.BLINK_TICK_MS / (1000 / FPS))
        self.tick_count = 0
        self.count = 0

    def consume(self, data):
        ear = data["ear"]
        out = None

        # check to see if the eye aspect ratio is below the blink
        # threshold
        if ear < EYE_AR_THRESH:
            blink = True
        else:
            blink = False

        if not self.count and blink:
            self.count += 1
            return out

        if self.count and blink:
            self.count += 1
            if not (self.count % self.num_planes_timeout) and self.tick_count < 2:
                self.tick_count += 1
                print("beep! {}, {}".format(self.count, self.tick_count))
                out = {"beeps": "tap"}
            return out


        if self.count and not blink:
            if self.tick_count:
                if self.tick_count == 1:
                    print("click! {}, {}".format(self.count, self.tick_count))
                    out = {"input_events": InputEvents.SKIP, "beeps": "click"}
                elif self.tick_count > 1:
                    print("click! {}, {}".format(self.count, self.tick_count))
                    out = {"input_events": InputEvents.ENTER, "beeps": "click"}

        self.count = 0
        self.tick_count = 0
        return out

class Beeper(Element):
    def __init__(self, name):
        Element.__init__(self, name, ElementType.SINK)

    def prepare(self):
        self.AUDIO_CHUNK = 2048
        self.tap = wave.open("../tap.wav", "rb")
        self.click = wave.open("../click.wav", "rb")
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.tap.getsampwidth()),
            channels = self.tap.getnchannels(),
            rate = self.tap.getframerate(),
            output = True,
            output_device_index = 0)

    def play(self, wave):
        samples = wave.readframes(self.AUDIO_CHUNK)
        while samples:
            self.stream.write(samples)
            samples = wave.readframes(self.AUDIO_CHUNK)
        wave.rewind()

    def consume(self, data):
        if data == "tap":
            self.play(self.tap)
        elif data == "click":
            self.play(self.click)

    def cleanup(self):
        self.tap.close()
        self.click.close()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class Main(Element):
    def __init__(self, name):
        Element.__init__(self, name, ElementType.SINK)

    def prepare(self):
        # We have to do this here, as opposed to on top of the module, since
        # inluding espeak creates some kind of runtime which is not available
        # (or so it seems) in the element's process context.
        from espeak import espeak
        self.sp = espeak.core
        self.sp.set_voice("spanish")

    def consume(self, data):
        if data == InputEvents.SKIP:
            self.sp.synth("si")
        elif data == InputEvents.ENTER:
            self.sp.synth("no")

def sigint_handler(signum, frame):
    pipe.stop()
    print("Bye bye!")

signal.signal(signal.SIGINT, sigint_handler)

pipe = Pipeline()
camera = [Camera("camera")]
blink = [BlinkDetector("blink_detector") for x in range(4)]
plot = [BlinkPlot("blink_plot")]
camera_plot = [CameraPlot("camera_plot")]
input_filter = [InputEventFilter("input_event_filter")]
beeper = [Beeper("beeper")]
main = [Main("main")]
pipe.link(camera, blink, queue_size=1)
pipe.link(blink, plot)
pipe.link(blink, camera_plot)
pipe.link(blink, input_filter)
pipe.link(input_filter, beeper, source_pad_name="beeps")
pipe.link(input_filter, main, source_pad_name="input_events")
pipe.start()

