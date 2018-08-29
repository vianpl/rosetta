# -*- coding: utf-8 -*-
# Blink
# TODO: add exceptions...
from multiprocessing import Process, Value, Pipe, Lock
from multiprocessing.connection import wait, Connection
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

class Pad(Connection):
    @classmethod
    def cast(cls, conn: Connection, name, direction, source_name, sink_name, lock, size=0):
       conn.__class__ = cls
       conn.name = name
       conn.direction = direction
       conn.source_name = source_name
       conn.sink_name = sink_name
       conn.lock = lock
       conn.size = size
       return conn

    def send(self, data):
        with self.lock:
            Connection.send(self, data)

    def recv(self):
        with self.lock:
            data = Connection.recv(self)

        return data

class Element(Process):
    def __init__(self, name, type):
        Process.__init__(self, name=name)
        self.type = type
        self.in_pads = []
        self.out_pads = []

    def add_pad(self, pad):
        if pad.direction is Direction.IN:
            self.in_pads.append(pad)
        elif pad.direction is Direction.OUT:
            self.out_pads.append(pad)
        else:
            print("Failed to add pad, wrong direction")

    def consume(self, data):
        return NotImplemented

    # optional
    def cleanup(self):
        pass

    # optional
    def prepare(self):
        pass

    def run(self):

        self.prepare()
        while self.status is PipelineStatus.PLAY:
            data = {}

            # Get data if relevant
            if self.type is not ElementType.SOURCE:
                for pad in wait(self.in_pads):
                    pad_data = pad.recv()
                    data[pad.name] = pad_data

            # Send it to children element
            if len(self.in_pads) == 1:
                data = self.consume(data[self.in_pads[0].name])
            else:
                data = self.consume(data)

            # Chain it to the next element if relevant
            if self.type is not ElementType.SINK and data is not None:
                for pad in self.out_pads:
                    if pad.name in data:
                        pad.send(data[pad.name])

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

    def link(self, sources, sinks, source_pad_name=None, sink_pad_name=None, queue_size=0):
        r, w = Pipe(duplex=False)
        send_lock = Lock()
        recv_lock = Lock()
        r = Pad.cast(r, sink_pad_name, Direction.IN, sources[0].name, sinks[0].name, recv_lock)
        w = Pad.cast(w, source_pad_name, Direction.OUT, sources[0].name, sinks[0].name, send_lock)

        for source in sources:
            source.add_pad(w)
            if not any(elem is source for elem in self.elements):
                self.elements.append(source)

        for sink in sinks:
            sink.add_pad(r)
            if not any(elem is sink for elem in self.elements):
                self.elements.append(sink)

    def start(self):
        for elem in self.elements:
            elem.start()

    def stop(self):
        for elem in self.elements:
            elem.stop()

