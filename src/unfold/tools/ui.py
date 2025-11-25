
import ipywidgets as widgets
from IPython.display import display

def true_false_dd(description, default=False):
    return widgets.Dropdown(
        options=[('True', True), ('False', False)],
        value=default,
        description=description
    )

def groomed_and_closure(default_groomed=False, default_closure=False):
    groomed = true_false_dd("Groomed:", default_groomed)
    closure = true_false_dd("Closure test:", default_closure)
    display(groomed, closure)
    return groomed, closure
