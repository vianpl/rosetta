# -*- coding: utf-8 -*-
# Blink
from scipy.spatial import distance as dist
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from collections import deque
from imutils import face_utils
from pipeline import *
from enum import Enum
# TODO: get rid of dependency, don't like it
import imutils
import signal
import dlib
import cv2

EYE_AR_THRESH = 0.3

class InputEvents(Enum):
    CLICK = 1
    OPTION = 2
    CLOSE = 3

class BlinkPlot(Element):
    def __init__(self, name, fps=30):
        Element.__init__(self, name, ElementType.SINK)
        self.fps = fps

    def prepare(self):
        self.cbuf = deque(maxlen=int(10*self.fps))
        for i in range(self.cbuf.maxlen):
            self.cbuf.append(EYE_AR_THRESH)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set_ylim(auto=True)
        self.ax.axhline(EYE_AR_THRESH, color='r')
        self.line, = self.ax.plot(self.cbuf)
        self.count = 0

    def consume(self, data):
        self.cbuf.append(data)
        self.line.set_ydata(self.cbuf)

        self.count += 1
        if self.count % 30 == 0:
            self.ax.relim()
            self.ax.autoscale_view()

        plt.draw()
        plt.pause(0.0000001)
        return None

    # def consume(self, data):
        # self.queue.put(data)

    def cleanup():
        plt.close()
        self.p.join()

class BlinkDetector(Element):
    def __init__(self, name, show=False):
        Element.__init__(self, name, ElementType.SOURCE)
        self.show = show

        # start the video stream thread
        self.vs = cv2.VideoCapture(0)
        self.FRAME_RATE = self.vs.get(cv2.CAP_PROP_FPS)

    def prepare(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        if (self.show):
            cv2.startWindowThread()
            cv2.namedWindow("Frame")

    def __eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear


    def consume(self, data):
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        ret, frame = self.vs.read()
        if ret != True:
            print("frame grab error {}:".format(ret))
            return None

        frame = imutils.resize(frame, width=450)
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

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[self.lStart:self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        leftEAR = self.__eye_aspect_ratio(leftEye)
        rightEAR = self.__eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        if self.show:
            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)

        return ear

    def cleanup(self):
        cv2.destroyAllWindows()
        self.vs.stop()


# A click is a BLINK_TIME blink event
# An option is a double BLINK_TIME event
# A close event is a full window with closed eyes
class InputEventFilter(Element):
    def __init__(self, name, fps=30):
        Element.__init__(self, name, ElementType.PROCESSING)
        self.fps = fps

    def prepare(self):
        self.WINDOW_SIZE = self.fps
        self.BLINK_TIME = round(self.fps * 0.2)
        self.window_pointer = 0
        self.blink_pointer = 0
        self.processing_blink = False
        self.window = [False] * self.WINDOW_SIZE

    def clear_window(self):
        for i in range(len(self.window)):
            self.window[1] = 0

    def process_window(self):
        eyes_closed = True
        num_blinks = 0
        positives = 0
        negatives = 0
        #print("process_window: {}".format(self.window))
        for blink in self.window:
            if not blink:
                positives = 0
                negatives += 1
                eyes_closed = False
            else:
                positives += 1
                if positives == self.BLINK_TIME:
                    num_blinks += 1

        if eyes_closed:
            return InputEvents.CLOSE

        if num_blinks == 1:
            return InputEvents.CLICK

        if num_blinks == 2:
            return InputEvents.OPTION

        return None

    def consume(self, ear):
        # check to see if the eye aspect ratio is below the blink
        # threshold
        if ear < EYE_AR_THRESH:
            blink = True
        else:
            blink = False

        if self.window_pointer >= self.WINDOW_SIZE:
            self.window_pointer = 0

        self.window[self.window_pointer] = blink
        self.window_pointer += 1

        if self.processing_blink and (self.window_pointer == self.blink_pointer):
            return self.process_window()
            self.processing_blink = False

        if (blink and not self.processing_blink):
            self.blink_pointer = self.window_pointer
            self.processing_blink = True

        return None


class Main(Element):
    def __init__(self, name):
        Element.__init__(self, name, ElementType.SINK)

    def consume(self, input_event):
        print("Input Event: {}".format(input_event))

def sigint_handler(signum, frame):
    pipe.stop()
    print("Bye bye!")

signal.signal(signal.SIGINT, sigint_handler)

pipe = Pipeline()
blink = BlinkDetector("blink_detector", True)
plot = BlinkPlot("blink_plot", blink.FRAME_RATE)
event = InputEventFilter("input_event_filter")
main = Main("main")
pipe.link_elements(blink, event)
pipe.link_elements(blink, plot)
pipe.link_elements(event, main)
pipe.start()

