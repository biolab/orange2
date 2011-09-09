'''

#################
Plot3D (``owplot3D``)
#################

.. autoclass:: OrangeWidgets.plot.OWPlot3D
    
'''

import os
import time
from math import sin, cos, pi
import struct

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtOpenGL

from OWDlgs import OWChooseImageSizeDlg
from Orange.misc import deprecated_attribute

import orangeqt
from plot.owplotgui import OWPlotGUI
from owtheme import PlotTheme
from owplot import OWPlot
from owlegend import OWLegend, OWLegendItem, OWLegendTitle, OWLegendGradient
from owopenglrenderer import OWOpenGLRenderer
from owconstants import ZOOMING, PANNING

from OWColorPalette import ColorPaletteGenerator

import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.FULL_LOGGING = False
OpenGL.ERROR_ON_COPY = False
from OpenGL.GL import *
from OpenGL.GL.ARB.vertex_array_object import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from ctypes import c_void_p, c_char, c_char_p, POINTER

import numpy
from numpy import array, maximum
#numpy.seterr(all='raise')

try:
    from itertools import chain
    from itertools import izip as zip
except:
    pass

def enum(*sequential):
    enums = dict(zip(sequential, range(len(sequential))))
    enums['is_valid'] = lambda self, enum_value: enum_value < len(sequential)
    enums['to_str'] = lambda self, enum_value: sequential[enum_value]
    enums['__len__'] = lambda self: len(sequential)
    return type('Enum', (), enums)()

PlotState = enum('IDLE', 'DRAGGING_LEGEND', 'ROTATING', 'SCALING', 'SELECTING', 'PANNING')

Symbol = enum('RECT', 'TRIANGLE', 'DTRIANGLE', 'CIRCLE', 'LTRIANGLE',
              'DIAMOND', 'WEDGE', 'LWEDGE', 'CROSS', 'XCROSS')

from plot.primitives import get_symbol_geometry, clamp, normalize, GeometryType

class OWLegend3D(OWLegend):
    def set_symbol_geometry(self, symbol, geometry):
        if not hasattr(self, '_symbol_geometry'):
            self._symbol_geometry = {}
        self._symbol_geometry[symbol] = geometry

    def _draw_item_background(self, pos, item, color):
        rect = item.rect().normalized().adjusted(pos.x(), pos.y(), pos.x(), pos.y())
        self.widget.renderer.draw_rectangle(
            QVector3D(rect.left(), rect.top(), 0),
            QVector3D(rect.left(), rect.bottom(), 0),
            QVector3D(rect.right(), rect.bottom(), 0),
            QVector3D(rect.right(), rect.top(), 0),
            color=color)

    def _draw_symbol(self, pos, symbol):
        edges = self._symbol_geometry[symbol.symbol()]
        color = symbol.color()
        size = symbol.size() / 2
        for v0, v1 in zip(edges[::2], edges[1::2]):
            x0, y0 = v0.x(), v0.y()
            x1, y1 = v1.x(), v1.y()
            self.widget.renderer.draw_line(
                QVector3D(x0*size + pos.x(), -y0*size + pos.y(), 0),
                QVector3D(x1*size + pos.x(), -y1*size + pos.y(), 0),
                color=color)

    def _paint(self, widget):
        '''Does all the drawing itself.'''
        self.widget = widget
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        offset = QPointF(0, 15) # TODO

        for category in self.items:
            items = self.items[category]
            for item in items:
                if isinstance(item, OWLegendTitle):
                    pos = self.pos() + item.pos()
                    self._draw_item_background(pos, item.rect_item, widget._theme.background_color)

                    widget.qglColor(widget._theme.labels_color)
                    pos = self.pos() + item.pos() + item.text_item.pos() + offset
                    widget.renderText(pos.x(), pos.y(), item.text_item.toPlainText(), item.text_item.font())
                elif isinstance(item, OWLegendItem):
                    pos = self.pos() + item.pos()
                    self._draw_item_background(pos, item.rect_item, widget._theme.background_color)

                    widget.qglColor(widget._theme.labels_color)
                    pos = self.pos() + item.pos() + item.text_item.pos() + offset
                    widget.renderText(pos.x(), pos.y(), item.text_item.toPlainText(), item.text_item.font())

                    symbol = item.point_item
                    pos = self.pos() + item.pos() + symbol.pos()
                    self._draw_symbol(pos, symbol)
                elif isinstance(item, OWLegendGradient):
                    pos = self.pos() + item.pos()
                    proxy = lambda: None
                    proxy.rect = lambda: item.rect
                    self._draw_item_background(pos, proxy, widget._theme.background_color)

                    widget.qglColor(widget._theme.labels_color)
                    for label in item.label_items:
                        pos = self.pos() + item.pos() + label.pos() + offset + QPointF(5, 0)
                        widget.renderText(pos.x(), pos.y(), label.toPlainText(), label.font())

                    pos = self.pos() + item.pos() + item.gradient_item.pos()
                    rect = item.gradient_item.rect().normalized().adjusted(pos.x(), pos.y(), pos.x(), pos.y())
                    widget.renderer.draw_rectangle(
                        QVector3D(rect.left(), rect.top(), 0),
                        QVector3D(rect.left(), rect.bottom(), 0),
                        QVector3D(rect.right(), rect.bottom(), 0),
                        QVector3D(rect.right(), rect.top(), 0),
                        QColor(0, 0, 0),
                        QColor(0, 0, 255),
                        QColor(0, 0, 255),
                        QColor(0, 0, 0))

class OWPlot3D(orangeqt.Plot3D):
    '''
    The base class behind 3D plots in Orange. Uses OpenGL as its rendering platform.

    **Plot layout**

        .. attribute:: show_legend
    
            A boolean controlling whether the legend is displayed or not

    **Settings**
    
	.. attribute:: gui
	
            An :obj:`.OWPlotGUI` object associated with this graph

    **Data**
        This is the most important part of the class API. :meth:`set_plot_data` is
        used to (not surprisingly) set the data which will be drawn.
        :meth:`set_features` tells Plot3D how to interpret the data (this method must
        be called after :meth:`set_plot_data` and can be called multiple times).
        :meth:`set_valid_data` optionally informs the plot which examples are invalid and
        should not be drawn. It should be called after set_plot_data, but before set_features.

        .. automethod:: set_plot_data

        .. automethod:: set_valid_data

        .. automethod:: set_features

    **Selections**

        .. method: get_selected_indices

        .. method: get_min_max_selected

        .. method: set_selection_behavior

        .. method: unselect_all_points

    **Callbacks**

        Plot3D provides several callbacks which can be used to perform additional tasks (
        such as drawing geometry before the data is drawn). Callback provided:

        auto_send_selection_callback
        mouseover_callback
        before_draw_callback
        after_draw_callback

    **Coordinate transformations**

        .. automethod:: map_to_plot

        .. automethod:: map_to_data

    **Colors**

        Colors are specified with two palettes, one for continuous attributes, and one for
        discrete ones.  Both are created by
        :obj:`.OWColorPalette.ColorPaletteGenerator`
        
        .. attribute:: continuous_palette
        
            The palette used when point color represents a continuous attribute
        
        .. attribute:: discrete_palette
        
            The palette used when point color represents a discrete attribute

    '''
    def __init__(self, parent=None):
        orangeqt.Plot3D.__init__(self, parent)

        # Don't clear background when using QPainter
        self.setAutoFillBackground(False)

        self.camera_distance = 6.

        self.scale_factor = 0.30
        self.rotation_factor = 0.3
        self.zoom_factor = 2000.

        self.yaw = self.pitch = -pi / 4.
        self.panning_factor = 0.8
        self.update_camera()

        self.perspective_near = 0.5
        self.perspective_far = 10.
        self.camera_fov = 14.

        self.show_legend = True
        self._legend = OWLegend3D(self, None)
        self._legend_margin = QRectF(0, 0, 100, 0)
        self._legend_moved = False
        self._legend.set_floating(True)
        self._legend.set_orientation(Qt.Vertical)

        self.use_2d_symbols = False
        self.symbol_scale = 1.
        self.alpha_value = 255
        self._zoomed_size = [1., 1., 1.]

        self._state = PlotState.IDLE

        self._selection = None
        self.selection_behavior = OWPlot.AddSelection

        ## Callbacks
        self.auto_send_selection_callback = None
        self.mouseover_callback = None
        self.before_draw_callback = None
        self.after_draw_callback = None

        self.setMouseTracking(True)
        self._mouse_position = QPoint(0, 0)
        self.invert_mouse_x = False
        self.mouse_sensitivity = 5

        self.additional_scale = array([0., 0., 0.])
        self.data_scale = array([1., 1., 1.])
        self.data_translation = array([0., 0., 0.])
        self.plot_scale = array([1., 1., 1.])
        self.plot_translation = -array([0.5, 0.5, 0.5])

        self._zoom_stack = []

        self._theme = PlotTheme()

        self._tooltip_fbo_dirty = True
        self._tooltip_win_center = [0, 0]

        self._use_fbos = True

        # If True, do drawing using instancing + geometry shader processing,
        # if False, build VBO every time set_plot_data is called.
        self._use_opengl_3 = False

        self.hide_outside = False
        self.fade_outside = True
        self.label_index = -1

        self.data = None

        self.continuous_palette = ColorPaletteGenerator(numberOfColors=-1)
        self.discrete_palette = ColorPaletteGenerator()

        self.gui = OWPlotGUI(self)

    def __del__(self):
        pass
        # TODO: never reached!
        #glDeleteVertexArrays(1, self.dummy_vao)
        #glDeleteVertexArrays(1, self.feedback_vao)
        #glDeleteBuffers(1, self.symbol_buffer)
        #if hasattr(self, 'data_buffer'):
        #    glDeleteBuffers(1, self.data_buffer)

    def legend(self):
        return self._legend

    def initializeGL(self):
        if hasattr(self, '_init_done'):
            return
        self.makeCurrent()
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_CULL_FACE)
        glEnable(GL_MULTISAMPLE)

        # TODO: check hardware for OpenGL 3.x+ support

        self.renderer = OWOpenGLRenderer()

        if self._use_opengl_3:
            self.feedback_generated = False

            self.generating_program = QtOpenGL.QGLShaderProgram()
            self.generating_program.addShaderFromSourceFile(QtOpenGL.QGLShader.Geometry,
                os.path.join(os.path.dirname(__file__), 'generator.gs'))
            self.generating_program.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex,
                os.path.join(os.path.dirname(__file__), 'generator.vs'))
            varyings = (c_char_p * 5)()
            varyings[:] = ['out_position', 'out_offset', 'out_color', 'out_normal', 'out_index']
            glTransformFeedbackVaryings(self.generating_program.programId(), 5, 
                ctypes.cast(varyings, POINTER(POINTER(c_char))), GL_INTERLEAVED_ATTRIBS)

            self.generating_program.bindAttributeLocation('index', 0)

            if not self.generating_program.link():
                print('Failed to link generating shader! Attribute changes may be slow.')
            else:
                print('Generating shader linked.')

            # Upload all symbol geometry into a TBO (texture buffer object), so that generating
            # geometry shader will have access to it. (TBO is easier to use than a texture in this use case).
            geometry_data = []
            symbols_indices = []
            symbols_sizes = []
            for symbol in range(len(Symbol)):
                triangles = get_symbol_geometry(symbol, GeometryType.SOLID_3D)
                symbols_indices.append(len(geometry_data) / 3)
                symbols_sizes.append(len(triangles))
                for tri in triangles:
                    geometry_data.extend(chain(*tri))

            for symbol in range(len(Symbol)):
                triangles = get_symbol_geometry(symbol, GeometryType.SOLID_2D)
                symbols_indices.append(len(geometry_data) / 3)
                symbols_sizes.append(len(triangles))
                for tri in triangles:
                    geometry_data.extend(chain(*tri))

            self.symbols_indices = symbols_indices
            self.symbols_sizes = symbols_sizes

            tbo = glGenBuffers(1)
            glBindBuffer(GL_TEXTURE_BUFFER, tbo)
            glBufferData(GL_TEXTURE_BUFFER, len(geometry_data)*4, numpy.array(geometry_data, 'f'), GL_STATIC_DRAW)
            glBindBuffer(GL_TEXTURE_BUFFER, 0)
            self.symbol_buffer = glGenTextures(1)
            glBindTexture(GL_TEXTURE_BUFFER, self.symbol_buffer)
            glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo) # 3 floating-point components
            glBindTexture(GL_TEXTURE_BUFFER, 0)

            # Generate dummy vertex buffer (points which will be fed to the geometry shader).
            self.dummy_vao = GLuint(0)
            glGenVertexArrays(1, self.dummy_vao)
            glBindVertexArray(self.dummy_vao)
            vertex_buffer_id = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
            glBufferData(GL_ARRAY_BUFFER, numpy.arange(50*1000, dtype=numpy.float32), GL_STATIC_DRAW)
            glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4, c_void_p(0))
            glEnableVertexAttribArray(0)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            # Specify an output VBO (and VAO)
            self.feedback_vao = feedback_vao = GLuint(0)
            glGenVertexArrays(1, feedback_vao)
            glBindVertexArray(feedback_vao)
            self.feedback_bid = feedback_bid = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, feedback_bid)
            vertex_size = (3+3+3+3+1)*4
            glBufferData(GL_ARRAY_BUFFER, 20*1000*144*vertex_size, c_void_p(0), GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(0))
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(3*4))
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(6*4))
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(9*4))
            glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(12*4))
            glEnableVertexAttribArray(0)
            glEnableVertexAttribArray(1)
            glEnableVertexAttribArray(2)
            glEnableVertexAttribArray(3)
            glEnableVertexAttribArray(4)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        else:
            # Load symbol geometry and send it to the C++ parent.
            geometry_data = []
            for symbol in range(len(Symbol)):
                triangles = get_symbol_geometry(symbol, GeometryType.SOLID_2D)
                triangles = [QVector3D(*v) for triangle in triangles for v in triangle]
                orangeqt.Plot3D.set_symbol_geometry(self, symbol, 0, triangles)

                triangles = get_symbol_geometry(symbol, GeometryType.SOLID_3D)
                triangles = [QVector3D(*v) for triangle in triangles for v in triangle]
                orangeqt.Plot3D.set_symbol_geometry(self, symbol, 1, triangles)

                edges = get_symbol_geometry(symbol, GeometryType.EDGE_2D)
                edges = [QVector3D(*v) for edge in edges for v in edge]
                orangeqt.Plot3D.set_symbol_geometry(self, symbol, 2, edges)
                self._legend.set_symbol_geometry(symbol, edges)

                edges = get_symbol_geometry(symbol, GeometryType.EDGE_3D)
                edges = [QVector3D(*v) for edge in edges for v in edge]
                orangeqt.Plot3D.set_symbol_geometry(self, symbol, 3, edges)

        self.symbol_program = QtOpenGL.QGLShaderProgram()
        self.symbol_program.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex,
            os.path.join(os.path.dirname(__file__), 'symbol.vs'))
        self.symbol_program.addShaderFromSourceFile(QtOpenGL.QGLShader.Fragment,
            os.path.join(os.path.dirname(__file__), 'symbol.fs'))

        self.symbol_program.bindAttributeLocation('position', 0)
        self.symbol_program.bindAttributeLocation('offset',   1)
        self.symbol_program.bindAttributeLocation('color',    2)
        self.symbol_program.bindAttributeLocation('normal',   3)
        self.symbol_program.bindAttributeLocation('index',    4)

        if not self.symbol_program.link():
            print('Failed to link symbol shader!')
        else:
            print('Symbol shader linked.')

        self.symbol_program_use_2d_symbols = self.symbol_program.uniformLocation('use_2d_symbols')
        self.symbol_program_symbol_scale   = self.symbol_program.uniformLocation('symbol_scale')
        self.symbol_program_alpha_value    = self.symbol_program.uniformLocation('alpha_value')
        self.symbol_program_scale          = self.symbol_program.uniformLocation('scale')
        self.symbol_program_translation    = self.symbol_program.uniformLocation('translation')
        self.symbol_program_hide_outside   = self.symbol_program.uniformLocation('hide_outside')
        self.symbol_program_fade_outside   = self.symbol_program.uniformLocation('fade_outside')
        self.symbol_program_force_color    = self.symbol_program.uniformLocation('force_color')
        self.symbol_program_encode_color   = self.symbol_program.uniformLocation('encode_color')

        format = QtOpenGL.QGLFramebufferObjectFormat()
        format.setAttachment(QtOpenGL.QGLFramebufferObject.Depth)
        self._tooltip_fbo = QtOpenGL.QGLFramebufferObject(256, 256, format)
        if self._tooltip_fbo.isValid():
            print('Tooltip FBO created.')
        else:
            print('Failed to create tooltip FBO! Tooltips disabled.')
            self._use_fbos = False

        self._init_done = True

    def resizeGL(self, width, height):
        pass

    def update_camera(self):
        self.pitch = clamp(self.pitch, -3., -0.1)
        self.camera = [
            sin(self.pitch)*cos(self.yaw),
            cos(self.pitch),
            sin(self.pitch)*sin(self.yaw)]

    def get_mvp(self):
        projection = QMatrix4x4()
        width, height = self.width(), self.height()
        aspect = float(width) / height if height != 0 else 1
        projection.perspective(self.camera_fov, aspect, self.perspective_near, self.perspective_far)

        modelview = QMatrix4x4()
        modelview.lookAt(
            QVector3D(self.camera[0]*self.camera_distance,
                      self.camera[1]*self.camera_distance,
                      self.camera[2]*self.camera_distance),
            QVector3D(0,-0.1, 0),
            QVector3D(0, 1, 0))

        return modelview, projection

    def paintEvent(self, event):
        glViewport(0, 0, self.width(), self.height())
        self.qglClearColor(self._theme.background_color)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        modelview, projection = self.get_mvp()
        self.modelview = modelview
        self.projection = projection

        if self.before_draw_callback:
            self.before_draw_callback()

        plot_scale = numpy.maximum([1e-5, 1e-5, 1e-5], self.plot_scale+self.additional_scale)

        self.symbol_program.bind()
        self.symbol_program.setUniformValue('modelview', self.modelview)
        self.symbol_program.setUniformValue('projection', self.projection)
        self.symbol_program.setUniformValue(self.symbol_program_use_2d_symbols, self.use_2d_symbols)
        self.symbol_program.setUniformValue(self.symbol_program_fade_outside,   self.fade_outside)
        self.symbol_program.setUniformValue(self.symbol_program_hide_outside,   self.hide_outside)
        self.symbol_program.setUniformValue(self.symbol_program_encode_color,   False)
        self.symbol_program.setUniformValue(self.symbol_program_symbol_scale,   self.symbol_scale, self.symbol_scale)
        self.symbol_program.setUniformValue(self.symbol_program_alpha_value,    self.alpha_value / 255., self.alpha_value / 255.)
        self.symbol_program.setUniformValue(self.symbol_program_scale,          *plot_scale)
        self.symbol_program.setUniformValue(self.symbol_program_translation,    *self.plot_translation)
        self.symbol_program.setUniformValue(self.symbol_program_force_color,    0., 0., 0., 0.)

        if self._use_opengl_3 and self.feedback_generated:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glBindVertexArray(self.feedback_vao)
            glDrawArrays(GL_TRIANGLES, 0, self.num_primitives_generated*3)
            glBindVertexArray(0)
        elif not self._use_opengl_3:
            glDisable(GL_CULL_FACE)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            orangeqt.Plot3D.draw_data(self, self.symbol_program.programId(), self.alpha_value / 255.)

        self.symbol_program.release()

        self.draw_labels()

        if self.after_draw_callback:
            self.after_draw_callback()

        if self._tooltip_fbo_dirty:
            self._tooltip_fbo.bind()
            glClearColor(1, 1, 1, 1)
            glClearDepth(1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDisable(GL_BLEND)
            glEnable(GL_DEPTH_TEST)

            glViewport(-self._mouse_position.x()+128, -(self.height()-self._mouse_position.y())+128, self.width(), self.height())
            self._tooltip_win_center = [self._mouse_position.x(), self._mouse_position.y()]

            self.symbol_program.bind()
            self.symbol_program.setUniformValue(self.symbol_program_encode_color, True)

            if self._use_opengl_3 and self.feedback_generated:
                glBindVertexArray(self.feedback_vao)
                glDrawArrays(GL_TRIANGLES, 0, self.num_primitives_generated*3)
                glBindVertexArray(0)
            elif not self._use_opengl_3:
                orangeqt.Plot3D.draw_data_solid(self)
            self.symbol_program.release()
            self._tooltip_fbo.release()
            self._tooltip_fbo_dirty = False
            glViewport(0, 0, self.width(), self.height())

        self.draw_helpers()

        if self.show_legend:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width(), self.height(), 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glDisable(GL_BLEND)

            self._legend._paint(self)

        self.swapBuffers()

    def draw_labels(self):
        if self.label_index < 0:
            return

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(array(self.projection.data(), dtype=float))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(array(self.modelview.data(), dtype=float))

        self.qglColor(self._theme.labels_color)
        for example in self.data.transpose():
            x = example[self.x_index]
            y = example[self.y_index]
            z = example[self.z_index]
            label = example[self.label_index]
            x, y, z = self.map_to_plot(array([x, y, z]), original=False)
            # TODO
            #if isinstance(label, str):
                #self.renderText(x,y,z, label, font=self._theme.labels_font)
            #else:
            self.renderText(x,y,z, ('%f' % label).rstrip('0').rstrip('.'),
                            font=self._theme.labels_font)

    def draw_helpers(self):
        glEnable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        projection = QMatrix4x4()
        projection.ortho(0, self.width(), self.height(), 0, -1, 1)
        modelview = QMatrix4x4()

        self.renderer.set_transform(projection, modelview)

        if self._state == PlotState.SCALING:
            x, y = self._mouse_position.x(), self._mouse_position.y()
            self.renderer.draw_triangle(QVector3D(x-5, y-30, 0),
                                        QVector3D(x+5, y-30, 0),
                                        QVector3D(x, y-40, 0),
                                        color=self._theme.helpers_color)
            self.renderer.draw_line(QVector3D(x, y, 0),
                                    QVector3D(x, y-30, 0),
                                    color=self._theme.helpers_color)
            self.renderer.draw_triangle(QVector3D(x-5, y-10, 0),
                                        QVector3D(x+5, y-10, 0),
                                        QVector3D(x, y, 0),
                                        color=self._theme.helpers_color)

            self.renderer.draw_triangle(QVector3D(x+10, y, 0),
                                        QVector3D(x+20, y-5, 0),
                                        QVector3D(x+20, y+5, 0),
                                        color=self._theme.helpers_color)
            self.renderer.draw_line(QVector3D(x+10, y, 0),
                                    QVector3D(x+40, y, 0),
                                    color=self._theme.helpers_color)
            self.renderer.draw_triangle(QVector3D(x+50, y, 0),
                                        QVector3D(x+40, y-5, 0),
                                        QVector3D(x+40, y+5, 0),
                                        color=self._theme.helpers_color)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width(), self.height(), 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            self.renderText(x, y-50, 'Scale y axis', font=self._theme.labels_font)
            self.renderText(x+60, y+3, 'Scale x and z axes', font=self._theme.labels_font)
        elif self._state == PlotState.SELECTING:
            internal_color = QColor(168, 202, 236, 50)
            self.renderer.draw_rectangle(QVector3D(self._selection.left(), self._selection.top(), 0),
                                         QVector3D(self._selection.right(), self._selection.top(), 0),
                                         QVector3D(self._selection.right(), self._selection.bottom(), 0),
                                         QVector3D(self._selection.left(), self._selection.bottom(), 0),
                                         internal_color,
                                         internal_color,
                                         internal_color,
                                         internal_color)

            border_color = QColor(51, 153, 255, 192)
            self.renderer.draw_line(QVector3D(self._selection.left(), self._selection.top(), 0),
                                    QVector3D(self._selection.right(), self._selection.top(), 0),
                                    border_color, border_color)
            self.renderer.draw_line(QVector3D(self._selection.right(), self._selection.top(), 0),
                                    QVector3D(self._selection.right(), self._selection.bottom(), 0),
                                    border_color, border_color)
            self.renderer.draw_line(QVector3D(self._selection.right(), self._selection.bottom(), 0),
                                    QVector3D(self._selection.left(), self._selection.bottom(), 0),
                                    border_color, border_color)
            self.renderer.draw_line(QVector3D(self._selection.left(), self._selection.bottom(), 0),
                                    QVector3D(self._selection.left(), self._selection.top(), 0),
                                    border_color, border_color)

    def set_features(self,
            x_index, y_index, z_index,
            color_index, symbol_index, size_index, label_index,
            colors, num_symbols_used,
            x_discrete, y_discrete, z_discrete,
            data_scale=array([1., 1., 1.]),
            data_translation=array([0., 0., 0.])):
        start = time.time()
        self.makeCurrent()
        self.data_scale = data_scale
        self.data_translation = data_translation
        self.x_index = x_index
        self.y_index = y_index
        self.z_index = z_index
        self.color_index = color_index
        self.symbol_index = symbol_index
        self.size_index = size_index
        self.colors = colors
        self.num_symbols_used = num_symbols_used
        self.x_discrete = x_discrete
        self.y_discrete = y_discrete
        self.z_discrete = z_discrete
        self.label_index = label_index

        # If color is a discrete attribute, colors should be a list of QColor

        if self._use_opengl_3:
            # Re-run generating program (geometry shader), store
            # results through transform feedback into a VBO on the GPU.
            self.generating_program.bind()
            self.generating_program.setUniformValue('x_index', x_index)
            self.generating_program.setUniformValue('y_index', y_index)
            self.generating_program.setUniformValue('z_index', z_index)
            #self.generating_program.setUniformValue('jitter_size', jitter_size)
            #self.generating_program.setUniformValue('jitter_continuous', jitter_continuous)
            self.generating_program.setUniformValue('x_discrete', x_discrete)
            self.generating_program.setUniformValue('y_discrete', y_discrete)
            self.generating_program.setUniformValue('z_discrete', z_discrete)
            self.generating_program.setUniformValue('color_index', color_index)
            self.generating_program.setUniformValue('symbol_index', symbol_index)
            self.generating_program.setUniformValue('size_index', size_index)
            self.generating_program.setUniformValue('use_2d_symbols', self.use_2d_symbols)
            self.generating_program.setUniformValue('example_size', self.example_size)
            self.generating_program.setUniformValue('num_colors', len(colors))
            self.generating_program.setUniformValue('num_symbols_used', num_symbols_used)
            # TODO: colors is list of QColor
            glUniform3fv(glGetUniformLocation(self.generating_program.programId(), 'colors'),
                len(colors), numpy.array(colors, 'f').ravel())
            glUniform1iv(glGetUniformLocation(self.generating_program.programId(), 'symbols_sizes'),
                len(Symbol)*2, numpy.array(self.symbols_sizes, dtype='i'))
            glUniform1iv(glGetUniformLocation(self.generating_program.programId(), 'symbols_indices'),
                len(Symbol)*2, numpy.array(self.symbols_indices, dtype='i'))

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_BUFFER, self.symbol_buffer)
            self.generating_program.setUniformValue('symbol_buffer', 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_BUFFER, self.data_buffer)
            self.generating_program.setUniformValue('data_buffer', 1)

            qid = glGenQueries(1)
            glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, qid)
            glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, self.feedback_bid)
            glEnable(GL_RASTERIZER_DISCARD)
            glBeginTransformFeedback(GL_TRIANGLES)

            glBindVertexArray(self.dummy_vao)
            glDrawArrays(GL_POINTS, 0, self.num_examples)

            glEndTransformFeedback()
            glDisable(GL_RASTERIZER_DISCARD)

            glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN)
            self.num_primitives_generated = glGetQueryObjectuiv(qid, GL_QUERY_RESULT)
            glBindVertexArray(0)
            self.feedback_generated = True
            print('Num generated primitives: ' + str(self.num_primitives_generated))

            self.generating_program.release()
            glActiveTexture(GL_TEXTURE0)
            print('Generation took ' + str(time.time()-start) + ' seconds')
        else:
            start = time.time()
            orangeqt.Plot3D.update_data(self, x_index, y_index, z_index,
                color_index, symbol_index, size_index, label_index,
                colors, num_symbols_used,
                x_discrete, y_discrete, z_discrete, self.use_2d_symbols)
            print('Data processing took ' + str(time.time() - start) + ' seconds')

        self.update()

    def set_plot_data(self, data, subset_data=None):
        self.makeCurrent()
        self.data = data
        self.data_array = numpy.array(data.transpose().flatten(), dtype=numpy.float32)
        self.example_size = len(data)
        self.num_examples = len(data[0])

        if self._use_opengl_3:
            tbo = glGenBuffers(1)
            glBindBuffer(GL_TEXTURE_BUFFER, tbo)
            glBufferData(GL_TEXTURE_BUFFER, len(self.data_array)*4, self.data_array, GL_STATIC_DRAW)
            glBindBuffer(GL_TEXTURE_BUFFER, 0)

            self.data_buffer = glGenTextures(1)
            glBindTexture(GL_TEXTURE_BUFFER, self.data_buffer)
            GL_R32F = 0x822E
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tbo)
            glBindTexture(GL_TEXTURE_BUFFER, 0)
        else:
            orangeqt.Plot3D.set_data(self, long(self.data_array.ctypes.data),
                                     self.num_examples,
                                     self.example_size)

    def set_valid_data(self, valid_data):
        self.valid_data = numpy.array(valid_data, dtype=bool) # QList<bool> is being a PITA
        orangeqt.Plot3D.set_valid_data(self, long(self.valid_data.ctypes.data))

    def set_new_zoom(self, x_min, x_max, y_min, y_max, z_min, z_max, plot_coordinates=False):
        '''Specifies new zoom in data or plot coordinates.'''
        self._zoom_stack.append((self.plot_scale, self.plot_translation))

        max = array([x_max, y_max, z_max]).copy()
        min = array([x_min, y_min, z_min]).copy()
        if not plot_coordinates:
            min -= self.data_translation
            min *= self.data_scale
            max -= self.data_translation
            max *= self.data_scale
        center = (max + min) / 2.
        new_translation = -array(center)
        # Avoid division by zero by adding a small value (this happens when zooming in
        # on elements with the same value of an attribute).
        self._zoomed_size = array(map(lambda i: i+1e-5 if i == 0 else i, max-min))
        new_scale = 1. / self._zoomed_size
        self._animate_new_scale_translation(new_scale, new_translation)

    def _animate_new_scale_translation(self, new_scale, new_translation, num_steps=10):
        translation_step = (new_translation - self.plot_translation) / float(num_steps)
        scale_step = (new_scale - self.plot_scale) / float(num_steps)
        # Animate zooming: translate first for a number of steps,
        # then scale. Make sure it doesn't take too long.
        start = time.time()
        for i in range(num_steps):
            if time.time() - start > 1.:
                self.plot_translation = new_translation
                break
            self.plot_translation = self.plot_translation + translation_step
            self.repaint()
        for i in range(num_steps):
            if time.time() - start > 1.:
                self.plot_scale = new_scale
                break
            self.plot_scale = self.plot_scale + scale_step
            self.repaint()

    def zoom_out(self):
        if len(self._zoom_stack) < 1:
            new_translation = -array([0.5, 0.5, 0.5])
            new_scale = array([1., 1., 1.])
        else:
            new_scale, new_translation = self._zoom_stack.pop()
        self._animate_new_scale_translation(new_scale, new_translation)
        self._zoomed_size = 1. / new_scale

    def save_to_file(self):
        size_dlg = OWChooseImageSizeDlg(self, [], parent=self)
        size_dlg.exec_()

    def save_to_file_direct(self, file_name, size=None):
        img = self.grabFrameBuffer()
        if size != None:
            img = img.scaled(size)
        return img.save(file_name)

    def map_to_plot(self, point, original=True):
        if original:
            point -= self.data_translation
            point *= self.data_scale
        point += self.plot_translation
        plot_scale = maximum([1e-5, 1e-5, 1e-5], self.plot_scale+self.additional_scale)
        point *= plot_scale
        return point

    def map_to_data(self, point, original=True):
        plot_scale = maximum([1e-5, 1e-5, 1e-5], self.plot_scale+self.additional_scale)
        point /= plot_scale
        point -= self.plot_translation
        if original:
            point /= self.data_scale
            point += self.data_translation
        return point

    def get_min_max_selected(self, area):
        viewport = [0, 0, self.width(), self.height()]
        area = [min(area.left(), area.right()), min(area.top(), area.bottom()), abs(area.width()), abs(area.height())]
        min_max = orangeqt.Plot3D.get_min_max_selected(self, area, self.projection * self.modelview,
                                                       viewport,
                                                       QVector3D(*self.plot_scale), QVector3D(*self.plot_translation))
        return min_max

    def get_selected_indices(self):
        return orangeqt.Plot3D.get_selected_indices(self)

    def unselect_all_points(self):
        orangeqt.Plot3D.unselect_all_points(self)
        orangeqt.Plot3D.update_data(self, self.x_index, self.y_index, self.z_index,
                                    self.color_index, self.symbol_index, self.size_index, self.label_index,
                                    self.colors, self.num_symbols_used,
                                    self.x_discrete, self.y_discrete, self.z_discrete, self.use_2d_symbols)
        self.update()

    def set_selection_behavior(self, behavior):
        self.selection_behavior = behavior

    def mousePressEvent(self, event):
        pos = self._mouse_position = event.pos()
        buttons = event.buttons()

        self._selection = None

        if buttons & Qt.LeftButton:
            legend_pos = self._legend.pos()
            lx, ly = legend_pos.x(), legend_pos.y()
            if self._legend.boundingRect().adjusted(lx, ly, lx, ly).contains(pos.x(), pos.y()):
                event.scenePos = lambda: QPointF(pos)
                self._legend.mousePressEvent(event)
                self.setCursor(Qt.ClosedHandCursor)
                self._state = PlotState.DRAGGING_LEGEND
            elif self.state == PANNING:
                self._state = PlotState.PANNING
            else:
                self._state = PlotState.SELECTING
                self._selection = QRect(pos.x(), pos.y(), 0, 0)
        elif buttons & Qt.RightButton:
            if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                self._state = PlotState.SCALING
                self.scaling_init_pos = self._mouse_position
                self.additional_scale = array([0., 0., 0.])
            else:
                self.zoom_out()
            self.update()
        elif buttons & Qt.MiddleButton:
            if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                self._state = PlotState.PANNING
            else:
                self._state = PlotState.ROTATING

    def _check_mouseover(self, pos):
        if self.mouseover_callback != None and self._state == PlotState.IDLE:
            if abs(pos.x() - self._tooltip_win_center[0]) > 100 or\
               abs(pos.y() - self._tooltip_win_center[1]) > 100:
                self._tooltip_fbo_dirty = True
                self.update()
            # Use pixel-color-picking to read example index under mouse cursor (also called ID rendering).
            self._tooltip_fbo.bind()
            value = glReadPixels(pos.x() - self._tooltip_win_center[0] + 128,
                                 self._tooltip_win_center[1] - pos.y() + 128,
                                 1, 1,
                                 GL_RGBA,
                                 GL_UNSIGNED_BYTE)
            self._tooltip_fbo.release()
            value = struct.unpack('I', value)[0]
            # Check if value is less than 4294967295 (
            # the highest 32-bit unsigned integer) which
            # corresponds to white background in color-picking buffer.
            if value < 4294967295:
                self.mouseover_callback(value)

    def mouseMoveEvent(self, event):
        pos = event.pos()

        self._check_mouseover(pos)

        dx = pos.x() - self._mouse_position.x()
        dy = pos.y() - self._mouse_position.y()

        if self.invert_mouse_x:
            dx = -dx

        if self._state == PlotState.SELECTING:
            self._selection.setBottomRight(pos)
        elif self._state == PlotState.DRAGGING_LEGEND:
            event.scenePos = lambda: QPointF(pos)
            self._legend.mouseMoveEvent(event)
        elif self._state == PlotState.ROTATING:
            self.yaw += (self.mouse_sensitivity / 5.) * dx / (self.rotation_factor*self.width())
            self.pitch += (self.mouse_sensitivity / 5.) * dy / (self.rotation_factor*self.height())
            self.update_camera()
        elif self._state == PlotState.PANNING:
            right_vec = normalize(numpy.cross(self.camera, [0, 1, 0]))
            up_vec = normalize(numpy.cross(right_vec, self.camera))
            right_vec[0] *= dx / (self.width() * self.plot_scale[0] * self.panning_factor)
            right_vec[2] *= dx / (self.width() * self.plot_scale[2] * self.panning_factor)
            right_vec[0] *= (self.mouse_sensitivity / 5.)
            right_vec[2] *= (self.mouse_sensitivity / 5.)
            up_scale = self.height() * self.plot_scale[1] * self.panning_factor
            self.plot_translation -= right_vec + up_vec * (dy / up_scale) * (self.mouse_sensitivity / 5.)
        elif self._state == PlotState.SCALING:
            dx = pos.x() - self.scaling_init_pos.x()
            dy = pos.y() - self.scaling_init_pos.y()
            dx /= self.scale_factor * self.width()
            dy /= self.scale_factor * self.height()
            dy /= float(self._zoomed_size[1])
            dx *= self.mouse_sensitivity / 5.
            dy *= self.mouse_sensitivity / 5.
            right_vec = normalize(numpy.cross(self.camera, [0, 1, 0]))
            self.additional_scale = [-dx * abs(right_vec[0]) / float(self._zoomed_size[0]),
                                     dy,
                                     -dx * abs(right_vec[2]) / float(self._zoomed_size[2])]
        elif self._state == PlotState.IDLE:
            legend_pos = self._legend.pos()
            lx, ly = legend_pos.x(), legend_pos.y()
            if self._legend.boundingRect().adjusted(lx, ly, lx, ly).contains(pos.x(), pos.y()):
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.unsetCursor()

        self._mouse_position = pos
        self.update()

    def mouseReleaseEvent(self, event):
        if self._state == PlotState.DRAGGING_LEGEND:
            self._legend.mouseReleaseEvent(event)
        if self._state == PlotState.SCALING:
            self.plot_scale = numpy.maximum([1e-5, 1e-5, 1e-5], self.plot_scale+self.additional_scale)
            self.additional_scale = array([0., 0., 0.])
            self._state = PlotState.IDLE
        elif self._state == PlotState.SELECTING:
            self._selection.setBottomRight(event.pos())
            if self.state == ZOOMING: # self.state is actually set by OWPlotGUI (different from self._state!)
                min_max = self.get_min_max_selected(self._selection)
                self.set_new_zoom(*min_max, plot_coordinates=True)
            else:
                area = self._selection
                viewport = [0, 0, self.width(), self.height()]
                area = [min(area.left(), area.right()), min(area.top(), area.bottom()), abs(area.width()), abs(area.height())]
                orangeqt.Plot3D.select_points(self, area, self.projection * self.modelview,
                                              viewport,
                                              QVector3D(*self.plot_scale), QVector3D(*self.plot_translation),
                                              self.selection_behavior)
                self.makeCurrent()
                orangeqt.Plot3D.update_data(self, self.x_index, self.y_index, self.z_index,
                                            self.color_index, self.symbol_index, self.size_index, self.label_index,
                                            self.colors, self.num_symbols_used,
                                            self.x_discrete, self.y_discrete, self.z_discrete, self.use_2d_symbols)

                if self.auto_send_selection_callback:
                    self.auto_send_selection_callback()

        self._tooltip_fbo_dirty = True
        self.unsetCursor()
        self._state = PlotState.IDLE
        self.update()

    def wheelEvent(self, event):
        if event.orientation() == Qt.Vertical:
            delta = 1 + event.delta() / self.zoom_factor
            self.plot_scale *= delta
            self._tooltip_fbo_dirty = True
            self.update()

    def notify_legend_moved(self, pos):
        self._legend.set_floating(True, pos)
        self._legend.set_orientation(Qt.Vertical)

    def get_theme(self):
        return self._theme

    def set_theme(self, value):
        self._theme = value
        self.update()

    theme = pyqtProperty(PlotTheme, get_theme, set_theme)

    def color(self, role, group = None):
        if group:
            return self.palette().color(group, role)
        else:
            return self.palette().color(role)

    def set_palette(self, p):
        self.setPalette(p)
        self.update()

    def show_tooltip(self, text):
        x, y = self._mouse_position.x(), self._mouse_position.y()
        QToolTip.showText(self.mapToGlobal(QPoint(x, y)), text, self, QRect(x-3, y-3, 6, 6))

    def clear(self):
        self._legend.clear()
        self.data_scale = array([1., 1., 1.])
        self.data_translation = array([0., 0., 0.])
        self._tooltip_fbo_dirty = True
        self.feedback_generated = False

    def clear_plot_transformations(self):
        self._zoom_stack = []
        self._zoomed_size = [1., 1., 1.]
        self.plot_translation = -array([0.5, 0.5, 0.5])
        self.plot_scale = array([1., 1., 1.])
        self.additional_scale = array([0., 0., 0.])

    contPalette = deprecated_attribute("contPalette", "continuous_palette")
    discPalette = deprecated_attribute("discPalette", "discrete_palette")

if __name__ == "__main__":
    # TODO
    pass
