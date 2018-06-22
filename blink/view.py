# -*- coding: utf-8 -*-
# Blink

class View():
    def __init__(self, name, parent = None):
        self.name = str(name)
        self.parent = parent
        self.children = list()

        if self.parent:
            self.parent.__insert_child(self)
            print("Inserted {} into {}".format(self.name, self.parent.name))

    def __str__(self):
        if self.parent:
            return  "{}/{}".format(self.parent, self.name)
        else:
            return "/{}".format(self.name)

    def __insert_child(self, view):
        self.children.append(view)

    def find_child(self, name):
        if self.name is name:
            return self

        if not self.children:
            return None

        for child in self.children:
            ret = child.find_child(name)
            if ret is not None:
                return ret

    def get_word(self):
        return NotImplemented

    def process_event(self, event):
        return NotImplemented

    def next_view(self):
        return NotImplemented
