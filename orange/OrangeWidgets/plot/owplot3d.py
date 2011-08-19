"""
    .. class:: OWPlot3D
        Base class for 3D plots.

    .. attribute:: show_legend
        Determines whether to display the legend or not.

    .. attribute:: use_ortho
        If False, perspective projection is used instead.

    .. method:: clear()
        Removes everything from the graph.
"""

import os
import sys
import time
from math import sin, cos, pi, floor, ceil, log10
import struct

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtOpenGL
from OWDlgs import OWChooseImageSizeDlg

import orange

import OpenGL
OpenGL.ERROR_CHECKING = False # Turned off for performance improvement.
OpenGL.ERROR_LOGGING = False
OpenGL.FULL_LOGGING = False
OpenGL.ERROR_ON_COPY = False  # TODO: enable this to check for unwanted copying (wrappers)
from OpenGL.GL import *
from OpenGL.GL.ARB.vertex_array_object import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from ctypes import c_void_p, c_char, c_char_p, POINTER

import numpy
from numpy import array, maximum
#numpy.seterr(all='raise') # Raises exceptions on invalid numerical operations.

try:
    from itertools import izip as zip
    from itertools import chain
except:
    pass

# TODO: modern opengl renderer
def draw_triangle(x0, y0, x1, y1, x2, y2):
    glBegin(GL_TRIANGLES)
    glVertex2f(x0, y0)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()

def draw_line(x0, y0, x1, y1):
    glBegin(GL_LINES)
    glVertex2f(x0, y0)
    glVertex2f(x1, y1)
    glEnd()

def plane_visible(plane, location):
    normal = normal_from_points(*plane[:3])
    loc_plane = normalize(plane[0] - location)
    if numpy.dot(normal, loc_plane) > 0:
        return False
    return True

def nicenum(x, round):
    expv = floor(log10(x))
    f = x / pow(10., expv)
    if round:
        if f < 1.5: nf = 1.
        elif f < 3.: nf = 2.
        elif f < 7.: nf = 5.
        else: nf = 10.
    else:
        if f <= 1.: nf = 1.
        elif f <= 2.: nf = 2.
        elif f <= 5.: nf = 5.
        else: nf = 10.
    return nf * pow(10., expv)

def loose_label(min_value, max_value, num_ticks):
    '''Algorithm by Paul S. Heckbert (Graphics Gems).
       Generates a list of "nice" values between min and max,
       given the number of ticks. Also returns the number
       of fractional digits to use.
    '''
    range = nicenum(max_value-min_value, False)
    d = nicenum(range / float(num_ticks-1), True)
    plot_min = floor(min_value / d) * d
    plot_max = ceil(max_value / d) * d
    num_frac = int(max(-floor(log10(d)), 0))
    return numpy.arange(plot_min, plot_max + 0.5*d, d), num_frac

def enum(*sequential):
    enums = dict(zip(sequential, range(len(sequential))))
    enums['is_valid'] = lambda self, enum_value: enum_value < len(sequential)
    enums['to_str'] = lambda self, enum_value: sequential[enum_value]
    enums['__len__'] = lambda self: len(sequential)
    return type('Enum', (), enums)()

# States the plot can be in:
# * idle: mostly doing nothing, rotations are not considered special
# * dragging legend: user has pressed left mouse button and is now dragging legend
#   to a more suitable location
# * scaling: user has pressed right mouse button, dragging it up and down
#   scales data in y-coordinate, dragging it right and left scales data
#   in current horizontal coordinate (x or z, depends on rotation)
PlotState = enum('IDLE', 'DRAGGING_LEGEND', 'ROTATING', 'SCALING', 'SELECTING', 'PANNING')

Symbol = enum('RECT', 'TRIANGLE', 'DTRIANGLE', 'CIRCLE', 'LTRIANGLE',
              'DIAMOND', 'WEDGE', 'LWEDGE', 'CROSS', 'XCROSS')

SelectionType = enum('ZOOM', 'RECTANGLE', 'POLYGON')

Axis = enum('X', 'Y', 'Z', 'CUSTOM')

from owprimitives3d import *

class Legend(object):
    def __init__(self, plot):
        self.border_color = [0.5, 0.5, 0.5, 1]
        self.border_thickness = 2
        self.position = [0, 0]
        self.size = [0, 0]
        self.items = []
        self.plot = plot
        self.symbol_scale = 6
        self.font = QFont('Helvetica', 9)
        self.metrics = QFontMetrics(self.font)

    def add_item(self, symbol, color, size, title):
        '''Adds an item to the legend.
           Symbol can be integer value or enum Symbol.
           Color should be RGBA. Size should be between 0 and 1.
        '''
        if not Symbol.is_valid(symbol):
            print('Legend: invalid symbol')
            return
        self.items.append([symbol, color, size, title])
        self.size[0] = max(self.metrics.width(item[3]) for item in self.items) + 40
        self.size[1] = len(self.items) * self.metrics.height() + 4

    def clear(self):
        self.items = []

    def draw(self):
        if not self.items:
            return

        x, y = self.position
        w, h = self.size
        t = self.border_thickness

        # Draw legend outline first.
        glDisable(GL_DEPTH_TEST)
        glColor4f(*self.border_color)
        glBegin(GL_QUADS)
        glVertex2f(x,   y)
        glVertex2f(x+w, y)
        glVertex2f(x+w, y+h)
        glVertex2f(x,   y+h)
        glEnd()

        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glVertex2f(x+t,   y+t)
        glVertex2f(x+w-t, y+t)
        glVertex2f(x+w-t, y+h-t)
        glVertex2f(x+t,   y+h-t)
        glEnd()

        item_pos_y = y + t + 13

        for symbol, color, size, text in self.items:
            glColor4f(*color)
            triangles = get_2d_symbol_data(symbol)
            glBegin(GL_TRIANGLES)
            for v0, v1, v2, _, _, _ in triangles:
                glVertex2f(x+v0[0]*self.symbol_scale*size+10, item_pos_y+v0[1]*self.symbol_scale*size-5)
                glVertex2f(x+v1[0]*self.symbol_scale*size+10, item_pos_y+v1[1]*self.symbol_scale*size-5)
                glVertex2f(x+v2[0]*self.symbol_scale*size+10, item_pos_y+v2[1]*self.symbol_scale*size-5)
            glEnd()
            self.plot.renderText(x+t+30, item_pos_y, text, font=self.font)
            item_pos_y += self.metrics.height()

    def contains(self, x, y):
        return self.position[0] <= x <= self.position[0]+self.size[0] and\
               self.position[1] <= y <= self.position[1]+self.size[1]

    def move(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy

class RectangleSelection(object):
    def __init__(self, plot, first_vertex):
        self.plot = plot
        self.first_vertex = first_vertex
        self.current_vertex = first_vertex

    def contains(self, x, y):
        x1, x2 = sorted([self.first_vertex[0], self.current_vertex[0]])
        y1, y2 = sorted([self.first_vertex[1], self.current_vertex[1]])
        if x1 <= x <= x2 and\
           y1 <= y <= y2:
            return True
        return False

    def move(self, dx, dy):
        self.first_vertex[0] += dx
        self.first_vertex[1] += dy
        self.current_vertex[0] += dx
        self.current_vertex[1] += dy

    def draw(self):
        v1, v2 = self.first_vertex, self.current_vertex
        glLineWidth(1)
        glColor4f(*self.plot.theme.helpers_color)
        draw_line(v1[0], v1[1], v1[0], v2[1])
        draw_line(v1[0], v2[1], v2[0], v2[1])
        draw_line(v2[0], v2[1], v2[0], v1[1])
        draw_line(v2[0], v1[1], v1[0], v1[1])

    def draw_mask(self):
        v1, v2 = self.first_vertex, self.current_vertex
        glBegin(GL_QUADS)
        glVertex2f(v1[0], v1[1])
        glVertex2f(v1[0], v2[1])
        glVertex2f(v2[0], v2[1])
        glVertex2f(v2[0], v1[1])
        glEnd()

    def valid(self):
        return self.first_vertex != self.current_vertex
        # TODO: drop if too small

class PolygonSelection(object):
    def __init__(self, plot, first_vertex):
        self.plot = plot
        self.vertices = [first_vertex]
        self.current_vertex = first_vertex
        self.first_vertex = first_vertex
        self.polygon = None

    def add_current_vertex(self):
        distance = (self.current_vertex[0]-self.first_vertex[0])**2
        distance += (self.current_vertex[1]-self.first_vertex[1])**2
        if distance < 10**2:
            self.vertices.append(self.first_vertex)
            self.polygon = QPolygon([QPoint(x, y) for (x, y) in self.vertices])
            return True
        else:
            if self.check_intersections():
                return True
            self.vertices.append(self.current_vertex)
            return False

    def check_intersections(self):
        if len(self.vertices) < 3:
            return False

        current_line = QLineF(self.current_vertex[0], self.current_vertex[1],
                              self.vertices[-1][0], self.vertices[-1][1])
        intersection = QPointF()
        v1 = self.vertices[0]
        for i, v2 in enumerate(self.vertices[1:-1]):
            line = QLineF(v1[0], v1[1],
                          v2[0], v2[1])
            if current_line.intersect(line, intersection) == QLineF.BoundedIntersection:
                self.current_vertex = [intersection.x(), intersection.y()]
                self.vertices = [self.current_vertex] + self.vertices[i+1:]
                self.vertices.append(self.current_vertex)
                self.polygon = QPolygon([QPoint(x, y) for (x, y) in self.vertices])
                return True
            v1 = v2
        return False

    def contains(self, x, y):
        if self.polygon == None:
            return False
        return self.polygon.containsPoint(QPoint(x, y), Qt.OddEvenFill)

    def move(self, dx, dy):
        self.vertices = [[x+dx, y+dy] for x,y in self.vertices]
        self.current_vertex[0] += dx
        self.current_vertex[1] += dy
        self.polygon.translate(dx, dy)

    def draw(self):
        glLineWidth(1)
        glColor4f(*self.plot.theme.helpers_color)
        if len(self.vertices) == 1:
            v1, v2 = self.vertices[0], self.current_vertex
            draw_line(v1[0], v1[1], v2[0], v2[1])
            return
        last_vertex = self.vertices[0]
        for vertex in self.vertices[1:]:
            v1, v2 = vertex, last_vertex
            draw_line(v1[0], v1[1], v2[0], v2[1])
            last_vertex = vertex

        v1, v2 = last_vertex, self.current_vertex
        draw_line(v1[0], v1[1], v2[0], v2[1])

    def draw_mask(self):
        if len(self.vertices) < 3:
            return
        v0 = self.vertices[0]
        for i in range(1, len(self.vertices)-1):
            vi = self.vertices[i]
            vj = self.vertices[i+1]
            draw_triangle(v0[0], v0[1],
                          vi[0], vi[1],
                          vj[0], vj[1])

class PlotTheme(object):
    def __init__(self):
        self.labels_font = QFont('Helvetice', 8)
        self.helper_font = self.labels_font
        self.helpers_color = [0., 0., 0., 1.]        # Color used for helping arrows when scaling.
        self.background_color = [1., 1., 1., 1.]     # Color in the background.
        self.axis_title_font = QFont('Helvetica', 10, QFont.Bold)
        self.axis_font = QFont('Helvetica', 9)
        self.labels_color = [0., 0., 0., 1.]
        self.axis_color = [0.1, 0.1, 0.1, 1.]
        self.axis_values_color = [0.1, 0.1, 0.1, 1.]

class OWPlot3D(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers), parent)

        self.camera_distance = 3.

        self.yaw = self.pitch = -pi / 4.
        self.rotation_factor = 0.3
        self.panning_factor = 0.4
        self.update_camera()

        self.ortho_scale = 900.
        self.ortho_near = -1
        self.ortho_far = 2000
        self.perspective_near = 0.1
        self.perspective_far = 2000
        self.camera_fov = 30.
        self.zoom_factor = 2000.

        self.use_ortho = False
        self.show_legend = True
        self.legend = Legend(self)

        self.use_2d_symbols = False
        self.symbol_scale = 1.
        self.transparency = 255
        self.zoomed_size = [1., 1., 1.]

        self.state = PlotState.IDLE

        self.selections = []
        self.selection_changed_callback = None
        self.selection_updated_callback = None
        self.selection_type = SelectionType.ZOOM
        self.new_selection = None

        self.setMouseTracking(True)
        self.mouseover_callback = None
        self.mouse_pos = QPoint(0, 0)
        self.before_draw_callback = None
        self.after_draw_callback = None

        self.x_axis_labels = None
        self.y_axis_labels = None
        self.z_axis_labels = None

        self.x_axis_title = ''
        self.y_axis_title = ''
        self.z_axis_title = ''

        self.show_x_axis_title = self.show_y_axis_title = self.show_z_axis_title = True

        self.scale_factor = 0.05
        self.additional_scale = array([0., 0., 0.])
        self.data_scale = array([1., 1., 1.])
        self.data_translation = array([0., 0., 0.])
        self.plot_scale = array([1., 1., 1.])
        self.plot_translation = -array([0.5, 0.5, 0.5])

        self.zoom_stack = []

        self._theme = PlotTheme()
        self.show_axes = True

        self.tooltip_fbo_dirty = True
        self.tooltip_win_center = [0, 0]
        self.selection_fbo_dirty = True

        self.use_fbos = True
        self.use_geometry_shader = True

        self.hide_outside = False

        self.build_axes()
        
        self.data = None

    def __del__(self):
        # TODO: never reached!
        glDeleteVertexArrays(1, self.dummy_vao)
        glDeleteVertexArrays(1, self.feedback_vao)
        glDeleteBuffers(1, self.symbol_buffer)
        if hasattr(self, 'data_buffer'):
            glDeleteBuffers(1, self.data_buffer)

    def initializeGL(self):
        if hasattr(self, 'generating_program'):
            return
        self.makeCurrent()
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_CULL_FACE)
        glEnable(GL_MULTISAMPLE)

        self.feedback_generated = False

        # Build shader program which will generate triangle data to be outputed
        # to the screen in subsequent frames. Geometry shader is the heart
        # of the process - it will produce actual symbol geometry out of dummy points.
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
            self.use_geometry_shader = False
        else:
            print('Generating shader linked.')

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
        self.symbol_program_transparency   = self.symbol_program.uniformLocation('transparency')
        self.symbol_program_scale          = self.symbol_program.uniformLocation('scale')
        self.symbol_program_translation    = self.symbol_program.uniformLocation('translation')
        self.symbol_program_hide_outside   = self.symbol_program.uniformLocation('hide_outside')
        self.symbol_program_force_color    = self.symbol_program.uniformLocation('force_color')
        self.symbol_program_encode_color   = self.symbol_program.uniformLocation('encode_color')

        # TODO: if not self.use_geometry_shader

        # Upload all symbol geometry into a TBO (texture buffer object), so that generating
        # geometry shader will have access to it. (TBO is easier to use than a texture in this use case).
        geometry_data = []
        symbols_indices = []
        symbols_sizes = []
        for symbol in range(len(Symbol)):
            triangles = get_2d_symbol_data(symbol)
            symbols_indices.append(len(geometry_data) / 3)
            symbols_sizes.append(len(triangles))
            for tri in triangles:
                geometry_data.extend(chain(*tri))

        for symbol in range(len(Symbol)):
            triangles = get_symbol_data(symbol)
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
           
        # Create two FBOs (framebuffer objects):
        # - one will be used together with stencil mask to find out which
        #   examples have been selected (in an efficient way)
        # - the other one will be used for tooltips (data rendered will have
        #   larger screen coverage so it will be easily pointed at)
        format = QtOpenGL.QGLFramebufferObjectFormat()
        format.setAttachment(QtOpenGL.QGLFramebufferObject.CombinedDepthStencil)
        self.selection_fbo = QtOpenGL.QGLFramebufferObject(1024, 1024, format)
        if self.selection_fbo.isValid():
            print('Selection FBO created.')
        else:
            print('Failed to create selection FBO! Selections may be slow.')
            self.use_fbos = False

        self.tooltip_fbo = QtOpenGL.QGLFramebufferObject(256, 256, format)
        if self.tooltip_fbo.isValid():
            print('Tooltip FBO created.')
        else:
            print('Failed to create tooltip FBO! Tooltips disabled.')
            self.use_fbos = False

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
        if self.use_ortho:
            projection.ortho(-width / self.ortho_scale,
                              width / self.ortho_scale,
                             -height / self.ortho_scale,
                              height / self.ortho_scale,
                             self.ortho_near,
                             self.ortho_far)
        else:
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

    def paintGL(self):
        glViewport(0, 0, self.width(), self.height())
        glClearColor(*self._theme.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self.feedback_generated:
            return

        modelview, projection = self.get_mvp()
        self.modelview = modelview
        self.projection = projection

        if self.before_draw_callback:
            self.before_draw_callback()

        if self.show_axes:
            self.draw_axes()

        self.symbol_program.bind()
        self.symbol_program.setUniformValue('modelview', modelview)
        self.symbol_program.setUniformValue('projection', projection)
        self.symbol_program.setUniformValue(self.symbol_program_use_2d_symbols, self.use_2d_symbols)
        self.symbol_program.setUniformValue(self.symbol_program_hide_outside,   self.hide_outside)
        self.symbol_program.setUniformValue(self.symbol_program_encode_color,   False)
        # Specifying float uniforms with vec2 because of a weird bug in PyQt
        self.symbol_program.setUniformValue(self.symbol_program_symbol_scale,   self.symbol_scale, self.symbol_scale)
        self.symbol_program.setUniformValue(self.symbol_program_transparency,   self.transparency / 255., self.transparency / 255.)
        plot_scale = numpy.maximum([1e-5, 1e-5, 1e-5],                          self.plot_scale+self.additional_scale)
        self.symbol_program.setUniformValue(self.symbol_program_scale,          *plot_scale)
        self.symbol_program.setUniformValue(self.symbol_program_translation,    *self.plot_translation)
        self.symbol_program.setUniformValue(self.symbol_program_force_color,    0., 0., 0., 0.)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.feedback_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_primitives_generated*3)
        glBindVertexArray(0)

        self.symbol_program.release()

        self.draw_labels()

        if self.after_draw_callback:
            self.after_draw_callback()

        if self.tooltip_fbo_dirty:
            self.tooltip_fbo.bind()
            glClearColor(1, 1, 1, 1)
            glClearDepth(1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDisable(GL_BLEND)
            glEnable(GL_DEPTH_TEST)

            glViewport(-self.mouse_pos.x()+128, -(self.height()-self.mouse_pos.y())+128, self.width(), self.height())
            self.tooltip_win_center = [self.mouse_pos.x(), self.mouse_pos.y()]

            self.symbol_program.bind()
            self.symbol_program.setUniformValue(self.symbol_program_encode_color, True)
            glBindVertexArray(self.feedback_vao)
            glDrawArrays(GL_TRIANGLES, 0, self.num_primitives_generated*3)
            glBindVertexArray(0)
            self.symbol_program.release()
            self.tooltip_fbo.release()
            self.tooltip_fbo_dirty = False
            glViewport(0, 0, self.width(), self.height())

        if self.selection_fbo_dirty:
            # TODO: use transform feedback instead
            self.selection_fbo.bind()
            glClearColor(1, 1, 1, 1)
            glClearStencil(0)
            glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

            self.symbol_program.bind()
            self.symbol_program.setUniformValue(self.symbol_program_encode_color, True)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            glBindVertexArray(self.feedback_vao)
            glDrawArrays(GL_TRIANGLES, 0, self.num_primitives_generated*3)
            glBindVertexArray(0)
            self.symbol_program.release()

            # Also draw stencil masks to the screen. No need to
            # write color or depth information as well, so we
            # disable those.
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width(), self.height(), 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
            glDepthMask(GL_FALSE)
            glStencilMask(0x01)
            glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT)
            glStencilFunc(GL_ALWAYS, 0, ~0)
            glEnable(GL_STENCIL_TEST)
            for selection in self.selections:
                selection.draw_mask()
            glDisable(GL_STENCIL_TEST)
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            glDepthMask(GL_TRUE)
            self.selection_fbo.release()
            self.selection_fbo_dirty = False

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        if self.show_legend:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width(), self.height(), 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            self.legend.draw()

        self.draw_helpers()

    def draw_labels(self):
        if self.label_index < 0:
            return

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(array(self.projection.data(), dtype=float))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(array(self.modelview.data(), dtype=float))

        glColor4f(*self._theme.labels_color)
        for example in self.data.transpose():
            x = example[self.x_index]
            y = example[self.y_index]
            z = example[self.z_index]
            label = example[self.label_index]
            x, y, z = self.map_to_plot(array([x, y, z]), original=False)
            #if isinstance(label, str):
                #self.renderText(x,y,z, label, font=self._theme.labels_font)
            #else:
            self.renderText(x,y,z, ('%f' % label).rstrip('0').rstrip('.'),
                            font=self._theme.labels_font)

    def draw_helpers(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if self.state == PlotState.SCALING:
            x, y = self.mouse_pos.x(), self.mouse_pos.y()
            #TODO: replace with an image
            glColor4f(*self._theme.helpers_color)
            draw_triangle(x-5, y-30, x+5, y-30, x, y-40)
            draw_line(x, y, x, y-30)
            draw_triangle(x-5, y-10, x+5, y-10, x, y)

            draw_triangle(x+10, y, x+20, y-5, x+20, y+5)
            draw_line(x+10, y, x+40, y)
            draw_triangle(x+50, y, x+40, y-5, x+40, y+5)

            self.renderText(x, y-50, 'Scale y axis', font=self._theme.labels_font)
            self.renderText(x+60, y+3, 'Scale x and z axes', font=self._theme.labels_font)
        elif self.state == PlotState.SELECTING and self.new_selection != None:
            self.new_selection.draw()

        for selection in self.selections:
            selection.draw()

    def build_axes(self):
        edge_half = 1. / 2.
        x_axis = [[-edge_half, -edge_half, -edge_half], [edge_half, -edge_half, -edge_half]]
        y_axis = [[-edge_half, -edge_half, -edge_half], [-edge_half, edge_half, -edge_half]]
        z_axis = [[-edge_half, -edge_half, -edge_half], [-edge_half, -edge_half, edge_half]]

        self.x_axis = x_axis = numpy.array(x_axis)
        self.y_axis = y_axis = numpy.array(y_axis)
        self.z_axis = z_axis = numpy.array(z_axis)

        self.unit_x = unit_x = numpy.array([1., 0., 0.])
        self.unit_y = unit_y = numpy.array([0., 1., 0.])
        self.unit_z = unit_z = numpy.array([0., 0., 1.])
 
        A = y_axis[1]
        B = y_axis[1] + unit_x
        C = x_axis[1]
        D = x_axis[0]

        E = A + unit_z
        F = B + unit_z
        G = C + unit_z
        H = D + unit_z

        self.axis_plane_xy = [A, B, C, D]
        self.axis_plane_yz = [A, D, H, E]
        self.axis_plane_xz = [D, C, G, H]

        self.axis_plane_xy_back = [H, G, F, E]
        self.axis_plane_yz_right = [B, F, G, C]
        self.axis_plane_xz_top = [E, F, B, A]

    def draw_axes(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.projection.data(), dtype=float))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.modelview.data(), dtype=float))

        def draw_axis(line):
            glColor4f(*self._theme.axis_color)
            glLineWidth(2)
            glBegin(GL_LINES)
            glVertex3f(*line[0])
            glVertex3f(*line[1])
            glEnd()

        def draw_discrete_axis_values(axis, coord_index, normal, axis_labels):
            start, end = axis.copy()
            start_value = self.map_to_data(start.copy())[coord_index]
            end_value = self.map_to_data(end.copy())[coord_index]
            length = end_value - start_value
            for i, label in enumerate(axis_labels):
                value = (i + 1) * 2
                if start_value <= value <= end_value:
                    position = start + (end-start)*((value-start_value) / length)
                    glBegin(GL_LINES)
                    glVertex3f(*(position))
                    glVertex3f(*(position+normal*0.03))
                    glEnd()
                    position += normal * 0.1
                    self.renderText(position[0],
                                    position[1],
                                    position[2],
                                    label, font=self._theme.labels_font)

        def draw_values(axis, coord_index, normal, axis_labels):
            glColor4f(*self._theme.axis_values_color)
            glLineWidth(1)
            if axis_labels != None:
                draw_discrete_axis_values(axis, coord_index, normal, axis_labels)
                return
            start, end = axis.copy()
            start_value = self.map_to_data(start.copy())[coord_index]
            end_value = self.map_to_data(end.copy())[coord_index]
            values, num_frac = loose_label(start_value, end_value, 7)
            for value in values:
                if not (start_value <= value <= end_value):
                    continue
                position = start + (end-start)*((value-start_value) / float(end_value-start_value))
                text = ('%%.%df' % num_frac) % value
                glBegin(GL_LINES)
                glVertex3f(*(position))
                glVertex3f(*(position+normal*0.03))
                glEnd()
                position += normal * 0.1
                self.renderText(position[0],
                                position[1],
                                position[2],
                                text, font=self._theme.axis_font)

        def draw_axis_title(axis, title, normal):
            middle = (axis[0] + axis[1]) / 2.
            middle += normal * 0.1 if axis[0][1] != axis[1][1] else normal * 0.2
            self.renderText(middle[0], middle[1], middle[2],
                            title,
                            font=self._theme.axis_title_font)

        glDisable(GL_DEPTH_TEST)
        glLineWidth(1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        cam_in_space = numpy.array([
          self.camera[0]*self.camera_distance,
          self.camera[1]*self.camera_distance,
          self.camera[2]*self.camera_distance
        ])

        planes = [self.axis_plane_xy, self.axis_plane_yz,
                  self.axis_plane_xy_back, self.axis_plane_yz_right]
        normals = [[numpy.array([0,-1, 0]), numpy.array([-1, 0, 0])],
                   [numpy.array([0, 0,-1]), numpy.array([ 0,-1, 0])],
                   [numpy.array([0,-1, 0]), numpy.array([-1, 0, 0])],
                   [numpy.array([0,-1, 0]), numpy.array([ 0, 0,-1])]]
        visible_planes = [plane_visible(plane, cam_in_space) for plane in planes]
        xz_visible = not plane_visible(self.axis_plane_xz, cam_in_space)

        if visible_planes[0 if xz_visible else 2]:
            draw_axis(self.x_axis)
            draw_values(self.x_axis, 0, numpy.array([0, 0, -1]), self.x_axis_labels)
            if self.show_x_axis_title:
                draw_axis_title(self.x_axis, self.x_axis_title, numpy.array([0, 0, -1]))
        elif visible_planes[2 if xz_visible else 0]:
            draw_axis(self.x_axis + self.unit_z)
            draw_values(self.x_axis + self.unit_z, 0, numpy.array([0, 0, 1]), self.x_axis_labels)
            if self.show_x_axis_title:
                draw_axis_title(self.x_axis + self.unit_z,
                                self.x_axis_title, numpy.array([0, 0, 1]))

        if visible_planes[1 if xz_visible else 3]:
            draw_axis(self.z_axis)
            draw_values(self.z_axis, 2, numpy.array([-1, 0, 0]), self.z_axis_labels)
            if self.show_z_axis_title:
                draw_axis_title(self.z_axis, self.z_axis_title, numpy.array([-1, 0, 0]))
        elif visible_planes[3 if xz_visible else 1]:
            draw_axis(self.z_axis + self.unit_x)
            draw_values(self.z_axis + self.unit_x, 2, numpy.array([1, 0, 0]), self.z_axis_labels)
            if self.show_z_axis_title:
                draw_axis_title(self.z_axis + self.unit_x, self.z_axis_title, numpy.array([1, 0, 0]))

        try:
            rightmost_visible = visible_planes[::-1].index(True)
        except ValueError:
            return
        if rightmost_visible == 0 and visible_planes[0] == True:
            rightmost_visible = 3
        y_axis_translated = [self.y_axis+self.unit_x,
                             self.y_axis+self.unit_x+self.unit_z,
                             self.y_axis+self.unit_z,
                             self.y_axis]
        normals = [numpy.array([1, 0, 0]),
                   numpy.array([0, 0, 1]),
                   numpy.array([-1,0, 0]),
                   numpy.array([0, 0,-1])]
        axis = y_axis_translated[rightmost_visible]
        normal = normals[rightmost_visible]
        draw_axis(axis)
        draw_values(axis, 1, normal, self.y_axis_labels)
        if self.show_y_axis_title:
            draw_axis_title(axis, self.y_axis_title, normal)

    def set_shown_attributes_indices(self, x_index, y_index, z_index,
            color_index, symbol_index, size_index, label_index,
            colors, num_symbols_used,
            x_discrete, y_discrete, z_discrete, jitter_size, jitter_continuous,
            data_scale=array([1., 1., 1.]), data_translation=array([0., 0., 0.])):
        start = time.time()
        self.makeCurrent()
        self.data_scale = data_scale
        self.data_translation = data_translation
        self.x_index = x_index
        self.y_index = y_index
        self.z_index = z_index
        self.label_index = label_index

        # If color is a discrete attribute, colors should be a list of colors
        # each specified with vec3 (RGB).

        # Re-run generating program (geometry shader), store
        # results through transform feedback into a VBO on the GPU.
        self.generating_program.bind()
        self.generating_program.setUniformValue('x_index', x_index)
        self.generating_program.setUniformValue('y_index', y_index)
        self.generating_program.setUniformValue('z_index', z_index)
        self.generating_program.setUniformValue('jitter_size', jitter_size)
        self.generating_program.setUniformValue('jitter_continuous', jitter_continuous)
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
        self.updateGL()

    def set_data(self, data, subset_data=None):
        self.makeCurrent()
        #if self.data != None:
            #TODO: glDeleteBuffers(1, self.data_buffer)
        start = time.time()

        data_array = numpy.array(data.transpose().flatten(), dtype='f')
        self.example_size = len(data)
        self.num_examples = len(data[0])
        self.data = data

        tbo = glGenBuffers(1)
        glBindBuffer(GL_TEXTURE_BUFFER, tbo)
        glBufferData(GL_TEXTURE_BUFFER, len(data_array)*4, data_array, GL_STATIC_DRAW)
        glBindBuffer(GL_TEXTURE_BUFFER, 0)

        self.data_buffer = glGenTextures(1)
        glBindTexture(GL_TEXTURE_BUFFER, self.data_buffer)
        GL_R32F = 0x822E
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tbo)
        glBindTexture(GL_TEXTURE_BUFFER, 0)

        print('Uploading data to GPU took ' + str(time.time()-start) + ' seconds')

    def set_axis_labels(self, axis_id, labels):
        '''labels should be a list of strings'''
        if Axis.is_valid(axis_id) and axis_id != Axis.CUSTOM:
            setattr(self, Axis.to_str(axis_id).lower() + '_axis_labels', labels)

    def set_axis_title(self, axis_id, title):
        if Axis.is_valid(axis_id) and axis_id != Axis.CUSTOM:
            setattr(self, Axis.to_str(axis_id).lower() + '_axis_title', title)

    def set_show_axis_title(self, axis_id, show):
        if Axis.is_valid(axis_id) and axis_id != Axis.CUSTOM:
            setattr(self, 'show_' + Axis.to_str(axis_id).lower() + '_axis_title', show)

    def set_new_zoom(self, x_min, x_max, y_min, y_max, z_min, z_max):
        '''Specifies new zoom in data coordinates.'''
        self.selections = []
        self.zoom_stack.append((self.plot_scale, self.plot_translation))

        max = array([x_max, y_max, z_max]).copy()
        min = array([x_min, y_min, z_min]).copy()
        min -= self.data_translation
        min *= self.data_scale
        max -= self.data_translation
        max *= self.data_scale
        center = (max + min) / 2.
        new_translation = -array(center)
        # Avoid division by zero by adding a small value (this happens when zooming in
        # on elements with the same value of an attribute).
        self.zoomed_size = array(map(lambda i: i+1e-5 if i == 0 else i, max-min))
        new_scale = 1. / self.zoomed_size
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
            self.updateGL()
        for i in range(num_steps):
            if time.time() - start > 1.:
                self.plot_scale = new_scale
                break
            self.plot_scale = self.plot_scale + scale_step
            self.updateGL()

    def zoom_out(self):
        if len(self.zoom_stack) < 1:
            new_translation = -array([0.5, 0.5, 0.5])
            new_scale = array([1., 1., 1.])
        else:
            new_scale, new_translation = self.zoom_stack.pop()
        self._animate_new_scale_translation(new_scale, new_translation)
        self.zoomed_size = 1. / new_scale

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

    def get_selection_indices(self):
        if len(self.selections) == 0:
            return []

        width, height = self.width(), self.height()
        if self.use_fbos and width <= 1024 and height <= 1024:
            self.selection_fbo_dirty = True
            self.updateGL()

            self.selection_fbo.bind()
            color_pixels = glReadPixels(0, 0,
                                        width, height,
                                        GL_RGBA,
                                        GL_UNSIGNED_BYTE)
            stencil_pixels = glReadPixels(0, 0,
                                          width, height,
                                          GL_STENCIL_INDEX,
                                          GL_FLOAT)
            self.selection_fbo.release()
            stencils = struct.unpack('f'*width*height, stencil_pixels)
            colors = struct.unpack('I'*width*height, color_pixels)
            indices = set([])
            for stencil, color in zip(stencils, colors):
                if stencil > 0. and color < 4294967295:
                    indices.add(color)

            return indices
        else:
            # Slower method (projects points manually and checks containments).
            modelview, projection = self.get_mvp()
            proj_model = projection * modelview
            viewport = [0, 0, width, height]

            def project(x, y, z):
                projected = proj_model * QVector4D(x, y, z, 1)
                projected /= projected.z()
                winx = viewport[0] + (1 + projected.x()) * viewport[2] / 2
                winy = viewport[1] + (1 + projected.y()) * viewport[3] / 2
                winy = height - winy
                return winx, winy

            indices = []
            for i, example in enumerate(self.data.transpose()):
                x = example[self.x_index]
                y = example[self.y_index]
                z = example[self.z_index]
                x, y, z = self.map_to_plot(array([x,y,z]).copy(), original=False)
                x_win, y_win = project(x, y, z)
                if any(sel.contains(x_win, y_win) for sel in self.selections):
                    indices.append(i)

            return indices

    def set_selection_type(self, type):
        if SelectionType.is_valid(type):
            self.selection_type = type

    def mousePressEvent(self, event):
        pos = self.mouse_pos = event.pos()
        buttons = event.buttons()

        if buttons & Qt.LeftButton:
            if self.show_legend and self.legend.contains(pos.x(), pos.y()):
                self.state = PlotState.DRAGGING_LEGEND
                self.new_selection = None
            else:
                if self.state == PlotState.SELECTING:
                    return
                for selection in self.selections:
                    if selection.contains(pos.x(), pos.y()):
                        self.state = PlotState.PANNING
                        self.dragged_selection = selection
                        return
                self.state = PlotState.SELECTING
                if self.selection_type == SelectionType.RECTANGLE or\
                   self.selection_type == SelectionType.ZOOM:
                    self.new_selection = RectangleSelection(self, [pos.x(), pos.y()])
        elif buttons & Qt.RightButton:
            if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                self.selections = []
                self.new_selection = None
                self.state = PlotState.SCALING
                self.scaling_init_pos = self.mouse_pos
                self.additional_scale = array([0., 0., 0.])
            else:
                self.zoom_out()
            self.updateGL()
        elif buttons & Qt.MiddleButton:
            self.state = PlotState.ROTATING
            self.selections = []
            self.new_selection = None

    def mouseMoveEvent(self, event):
        pos = event.pos()

        if self.mouseover_callback != None and self.state == PlotState.IDLE and\
            (not self.show_legend or not self.legend.contains(pos.x(), pos.y())):
            if abs(pos.x() - self.tooltip_win_center[0]) > 100 or\
               abs(pos.y() - self.tooltip_win_center[1]) > 100:
                self.tooltip_fbo_dirty = True
                self.updateGL()
            # Use pixel-color-picking to read example index under mouse cursor.
            self.tooltip_fbo.bind()
            value = glReadPixels(pos.x() - self.tooltip_win_center[0] + 128,
                                 self.tooltip_win_center[1] - pos.y() + 128,
                                 1, 1,
                                 GL_RGBA,
                                 GL_UNSIGNED_BYTE)
            self.tooltip_fbo.release()
            value = struct.unpack('I', value)[0]
            # Check if value is less than 4294967295 (
            # the highest 32-bit unsigned integer) which
            # corresponds to white background in color-picking buffer.
            if value < 4294967295:
                self.mouseover_callback(value)

        if self.state == PlotState.IDLE:
            if any(sel.contains(pos.x(), pos.y()) for sel in self.selections) or\
               (self.show_legend and self.legend.contains(pos.x(), pos.y())):
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.mouse_pos = pos
            return

        dx = pos.x() - self.mouse_pos.x()
        dy = pos.y() - self.mouse_pos.y()

        if self.state == PlotState.SELECTING and self.new_selection != None:
            self.new_selection.current_vertex = [pos.x(), pos.y()]
        elif self.state == PlotState.DRAGGING_LEGEND:
            self.legend.move(dx, dy)
        elif self.state == PlotState.ROTATING:
            if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                right_vec = normalize(numpy.cross(self.camera, [0, 1, 0]))
                up_vec = normalize(numpy.cross(right_vec, self.camera))
                right_vec[0] *= dx / (self.width() * self.plot_scale[0] * self.panning_factor)
                right_vec[2] *= dx / (self.width() * self.plot_scale[2] * self.panning_factor)
                up_scale = self.height()*self.plot_scale[1]*self.panning_factor
                self.plot_translation -= right_vec + up_vec*(dy / up_scale)
            else:
                self.yaw += dx / (self.rotation_factor*self.width())
                self.pitch += dy / (self.rotation_factor*self.height())
                self.update_camera()
        elif self.state == PlotState.SCALING:
            dx = pos.x() - self.scaling_init_pos.x()
            dy = pos.y() - self.scaling_init_pos.y()
            dx /= float(self.zoomed_size[0]) # TODO
            dy /= float(self.zoomed_size[1])
            dx /= self.scale_factor * self.width()
            dy /= self.scale_factor * self.height()
            self.additional_scale = [dx, dy, 0]
        elif self.state == PlotState.PANNING:
            self.dragged_selection.move(dx, dy)

        self.mouse_pos = pos
        self.updateGL()

    def mouseReleaseEvent(self, event):
        if self.state == PlotState.SELECTING and self.new_selection == None:
            self.new_selection = PolygonSelection(self, [event.pos().x(), event.pos().y()])
            return

        if self.state == PlotState.SCALING:
            self.plot_scale = numpy.maximum([1e-5, 1e-5, 1e-5], self.plot_scale+self.additional_scale)
            self.additional_scale = array([0., 0., 0.])
            self.state = PlotState.IDLE
        elif self.state == PlotState.SELECTING:
            if self.selection_type == SelectionType.POLYGON:
                last = self.new_selection.add_current_vertex()
                if last:
                    self.selections.append(self.new_selection)
                    self.selection_changed_callback() if self.selection_changed_callback else None
                    self.state = PlotState.IDLE
                    self.new_selection = None
            else:
                if self.new_selection.valid():
                    self.selections.append(self.new_selection)
                    self.updateGL()
                    self.selection_changed_callback() if self.selection_changed_callback else None
        elif self.state == PlotState.PANNING:
            self.selection_updated_callback() if self.selection_updated_callback else None

        if not (self.state == PlotState.SELECTING and self.selection_type == SelectionType.POLYGON):
            self.state = PlotState.IDLE
            self.tooltip_fbo_dirty = True
            self.new_selection = None

        self.updateGL()

    def wheelEvent(self, event):
        if event.orientation() == Qt.Vertical:
            self.selections = []
            delta = 1 + event.delta() / self.zoom_factor
            self.plot_scale *= delta
            self.tooltip_fbo_dirty = True
            self.updateGL()

    def remove_last_selection(self):
        if len(self.selections) > 0:
            self.selections.pop()
            self.updateGL()
            self.selection_changed_callback() if self.selection_changed_callback else None

    def remove_all_selections(self):
        self.selections = []
        if self.selection_changed_callback and self.selection_type != SelectionType.ZOOM:
            self.selection_changed_callback()
        self.updateGL()

    @pyqtProperty(PlotTheme)
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, theme):
        self._theme = theme
        self.updateGL()

    def show_tooltip(self, text):
        x, y = self.mouse_pos.x(), self.mouse_pos.y()
        QToolTip.showText(self.mapToGlobal(QPoint(x, y)), text, self, QRect(x-3, y-3, 6, 6))

    def clear(self):
        self.selections = []
        self.legend.clear()
        self.zoom_stack = []
        self.zoomed_size = [1., 1., 1.]
        self.plot_translation = -array([0.5, 0.5, 0.5])
        self.plot_scale = array([1., 1., 1.])
        self.additional_scale = array([0., 0., 0.])
        self.data_scale = array([1., 1., 1.])
        self.data_translation = array([0., 0., 0.])
        self.x_axis_labels = None
        self.y_axis_labels = None
        self.z_axis_labels = None
        self.tooltip_fbo_dirty = True
        self.selection_fbo_dirty = True

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWPlot3D()
    w.show()
 
    from random import random
    rand = lambda: random() - 0.5
    N = 100
    data = orange.ExampleTable("../doc/datasets/iris.tab")
    array, c, _ = data.toNumpyMA()
    import OWColorPalette
    palette = OWColorPalette.ColorPaletteHSV(len(data.domain.classVar.values))
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]
    colors = [palette[int(ex.getclass())] for ex in data]
    colors = [[c.red()/255., c.green()/255., c.blue()/255., 0.8] for c in colors]
    w.scatter(x, y, z, colors=colors)
    app.exec_()
