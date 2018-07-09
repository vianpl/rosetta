# -*- coding: utf-8 -*-
# Blink
from view import View

class WordSpellView(View):
    def get_word(self):
        for child in self.children:
            yield child.name
