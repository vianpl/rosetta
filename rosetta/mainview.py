# -*- coding: utf-8 -*-
# Blink
from view import View

class MainView(View):
    def get_word(self):
        for child in self.children:
            yield child.name

    def next_view(self):
        return self.find_child("palabras")
