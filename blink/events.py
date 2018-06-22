# -*- coding: utf-8 -*-
# Blink
from scipy.spatial import distance as dist
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from imutils import face_utils
from enum import Enum
import imutils
import dlib
import cv2

class InputEvents(Enum):
    click = 1
    option = 2
    close = 3

class Blink:
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.3

    def __init__(self, queue, show=False):
        self.show = show
        self.run = True
        self.queue = queue

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # start the video stream thread
        self.vs = cv2.VideoCapture(0)
        self.FRAME_RATE = self.vs.get(cv2.CAP_PROP_FPS)

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


    def start(self):
        blink = False
        ear = 0
        plot = []

        if (self.show):
            cv2.startWindowThread()
            cv2.namedWindow("Frame")

        while self.run:
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            ret, frame = self.vs.read()
            if ret != True:
                print("frame grab error {}:".format(ret))
                continue

            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.__eye_aspect_ratio(leftEye)
                rightEAR = self.__eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                if self.show:
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold
                if ear < self.EYE_AR_THRESH:
                    blink = True
                else:
                    blink = False

                plot.append(ear)
                if (len(plot) == 500):
                    plt.plot(plot)
                    plt.show()

                #print("blink = {}, ear = {}".format(blink, ear))
                self.queue.put(blink)

                if self.show:
                    # draw the total number of blinks on the frame along with
                    # the computed eye aspect ratio for the frame
                    cv2.putText(frame, "Blink: {}".format(blink), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # show the frame
            if self.show:
                cv2.imshow("Frame", frame)

    def stop(self):
        self.run = False
        cv2.destroyAllWindows()
        self.vs.stop()


# A click is a BLINK_TIME blink event
# An option is a double BLINK_TIME event
# A close event is a full window with closed eyes
class EventFilter:
    def __init__(self, queue, fps=30):
        self.WINDOW_SIZE = fps
        self.BLINK_TIME = round(fps * 0.2)
        self.in_queue = queue
        self.out_queue = Queue()
        self.window_pointer = 0
        self.blink_pointer = 0
        self.processing_blink = False
        self.window = [False] * self.WINDOW_SIZE

    def get_input(self):
        return self.out_queue.get()

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
            self.out_queue.put(InputEvents.close)

        if num_blinks == 1:
            self.out_queue.put(InputEvents.click)

        if num_blinks == 2:
            self.out_queue.put(InputEvents.option)

    def start(self):
        while True:
            blink = self.in_queue.get()
            if self.window_pointer >= self.WINDOW_SIZE:
                self.window_pointer = 0

            self.window[self.window_pointer] = blink
            self.window_pointer += 1

            if self.processing_blink and (self.window_pointer == self.blink_pointer):
                self.process_window()
                self.processing_blink = False

            if (blink and not self.processing_blink):
                self.blink_pointer = self.window_pointer
                self.processing_blink = True


def run():
    while True:
        input_event = event.get_input()
        print("Input Event: {}".format(input_event))


blink_queue = Queue()
blink = Blink(blink_queue, True)
event = EventFilter(blink_queue)

b = Process(target=blink.start)
e = Process(target=event.start)
m = Process(target=run)
b.start()
e.start()
m.start()
b.join()
e.join()
m.join()
