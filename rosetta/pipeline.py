# -*- coding: utf-8 -*-
# Blink
# TODO: add exceptions...
from multiprocessing import Process, Queue
from enum import Enum

class ElementType(Enum):
    SOURCE = 1
    SINK = 2
    PROCESSING = 3

class Direction(Enum):
    IN = 1
    OUT = 2

class PipelineStatus(Enum):
    STOP = 1
    PLAY = 2

class Pad:
    def __init__(self, queue, direction):
        self.queue = queue
        self.direction = direction

    def chain(self, data):
        self.queue.put(data)

    def get(self):
        return self.queue.get()

class Element(Process):
    def __init__(self, name, type):
        Process.__init__(self, name=name)
        self.type = type
        self.in_pad = None
        self.out_pads = []

    def set_pad(self, direction, queue):
        pad = Pad(queue, direction)
        if direction is Direction.IN:
            if self.in_pad is None:
                self.in_pad = pad
            else:
                print("Failed to add pad, {} already has an in pad".format(self.name))
        elif direction is Direction.OUT:
            self.out_pads.append(pad)
        else:
            print("Failed to add pad, wrong direction")

    def chain(self, data):
        if not self.out_pads:
            print("Can't chain {}, no pads".format(self.name))

        for pad in self.out_pads:
            pad.chain(data)

    def get(self):
        if self.in_pad is None:
            print("Can't get data from {}, no in pads".format(self.name))

        return self.in_pad.get()

    def consume(self, data):
        return NotImplemented

    # optional
    def cleanup(self):
        pass

    def prepare(self):
        pass

    def run(self):
        data = None

        self.prepare()
        while self.status is PipelineStatus.PLAY:
            if self.type is not ElementType.SOURCE:
                data = self.in_pad.get()

            #print("consume in {}".format(self.name))
            data = self.consume(data)
            #print("consume out {}, data {}".format(self.name, data))

            if self.type is not ElementType.SINK and data is not None:
                for pad in self.out_pads:
                    pad.chain(data)

        print("out {}".format(self.name))

        self.cleanup()

    def start(self):
        # TODO: This is wrong use Values from multiprocessing
        self.status = PipelineStatus.PLAY
        Process.start(self)

    def stop(self):
        self.status = PipelineStatus.STOP
        Process.join(self)

class Pipeline:
    def __init__(self):
        self.elements = []

    def link_elements(self, source, sink):
        queue = Queue()
        source.set_pad(Direction.OUT, queue)
        sink.set_pad(Direction.IN, queue)

        # Save elements if needed
        if not any(elem is source for elem in self.elements):
            self.elements.append(source)
        if not any(elem is sink for elem in self.elements):
            self.elements.append(sink)

    def start(self):
        for elem in self.elements:
            elem.start()

    def stop(self):
        for elem in self.elements:
            elem.stop()

