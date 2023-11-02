from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from tvtk.pyface.scene_model import SceneModel
from tvtk.pyface.scene_editor import SceneEditor

class MyModel(HasTraits):
    scene = Instance(SceneModel, ())

    view = View(Item('scene', height=400, show_label=False,
                    editor=SceneEditor()))

MyModel().configure_traits()