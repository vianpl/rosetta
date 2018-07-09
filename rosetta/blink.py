from enum import Enum
from view import View
from wordspellview import WordSpellView
from mainview import MainView

class Tts:
    def __init__(self):
        return NotImplemented

    def say(self):
        return NotImplemented

    def stop(self):
        return NotImplemented

class ExpressionDB:
    def __init__(self):
        return NotImplemented

    def get_expression(self, folder, index):
        return NotImplemented

    def get_folder(self, index):
        return NotImplemented
# folder iterator
# expresion iterator

print("initialization....")
main = MainView("main")
palabras = WordSpellView("palabras", main)
opt_palabras = View("opt_palabaras", palabras)
frecuentes = View("frecuentes", main)
print(opt_palabras)
print()

print("Starting....\n")
view = main
while True:
    for word in view.get_word():
        if word:
            print(word)
    print()
    view = view.next_view()


