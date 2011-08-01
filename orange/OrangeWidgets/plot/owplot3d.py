"""
    .. class:: OWPlot3D
        Base class for 3D plots.

    .. attribute:: show_legend
        Determines whether to display the legend or not.

    .. attribute:: use_ortho
        If False, perspective projection is used instead.

    .. method:: set_x_axis_title(title)
        Sets ``title`` as the current title (label) of x axis.

    .. method:: set_y_axis_title(title)
        Sets ``title`` as the current title (label) of y axis.

    .. method:: set_z_axis_title(title)
        Sets ``title`` as the current title (label) of z axis.

    .. method:: set_show_x_axis_title(show)
        Determines whether to show the title of x axis or not.

    .. method:: set_show_y_axis_title(show)
        Determines whether to show the title of y axis or not.

    .. method:: set_show_z_axis_title(show)
        Determines whether to show the title of z axis or not.

    .. method:: scatter(X, Y, Z, c, s)
        Adds scatter data to command buffer. ``X``, ``Y`` and ``Z`
        should be arrays (of equal length) with example data.
        ``c`` is optional, can be an array as well (setting
        colors of each example) or string ('r', 'g' or 'b'). ``s``
        optionally sets sizes of individual examples.

    .. method:: clear()
        Removes everything from the graph.
"""

# TODO: docs!

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtOpenGL
from OWDlgs import OWChooseImageSizeDlg

import orange

import OpenGL
OpenGL.ERROR_CHECKING = False # Turned off for performance improvement.
OpenGL.ERROR_LOGGING = False
OpenGL.FULL_LOGGING = False
#OpenGL.ERROR_ON_COPY = True  # TODO: enable this to check for unwanted copying (wrappers)
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_array_object import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from ctypes import c_void_p

import sys
from math import sin, cos, pi, floor, ceil, log10
import time
import struct
import numpy
#numpy.seterr(all='raise') # Raises exceptions on invalid numerical operations.

try:
    from itertools import izip as zip # Python 3 zip == izip in Python 2.x
except:
    pass

def normalize(vec):
    return vec / numpy.sqrt(numpy.sum(vec**2))

def clamp(value, min, max):
    if value < min:
        return min
    if value > max:
        return max
    return value

def normal_from_points(p1, p2, p3):
    if isinstance(p1, (list, tuple)):
        v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]
        v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]]
    else:
        v1 = p2 - p1
        v2 = p3 - p1
    return normalize(numpy.cross(v1, v2))

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

from owprimitives3d import get_symbol_data, get_2d_symbol_data, get_2d_symbol_edges

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
                glVertex2f(x+v0[0]*self.symbol_scale+10, item_pos_y+v0[1]*self.symbol_scale-5)
                glVertex2f(x+v1[0]*self.symbol_scale+10, item_pos_y+v1[1]*self.symbol_scale-5)
                glVertex2f(x+v2[0]*self.symbol_scale+10, item_pos_y+v2[1]*self.symbol_scale-5)
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
        self.axis_title_font = QFont('Helvetica', 10, QFont.Bold)
        self.axis_font = QFont('Helvetica', 9)
        self.helper_font = self.labels_font
        self.grid_color = [0.8, 0.8, 0.8, 1.]        # Color of the cube grid.
        self.labels_color = [0., 0., 0., 1.]         # Color used for example labels.
        self.helpers_color = [0., 0., 0., 1.]        # Color used for helping arrows when scaling.
        self.axis_color = [0.1, 0.1, 0.1, 1.]        # Color of the axis lines.
        self.axis_values_color = [0.1, 0.1, 0.1, 1.]
        self.background_color = [1., 1., 1., 1.]     # Color in the background.

class LightTheme(PlotTheme):
    pass

class DarkTheme(PlotTheme):
    def __init__(self):
        super(DarkTheme, self).__init__()
        self.grid_color = [0.3, 0.3, 0.3, 1.]
        self.labels_color = [0.9, 0.9, 0.9, 1.]
        self.helpers_color = [0.9, 0.9, 0.9, 1.]
        self.axis_values_color = [0.7, 0.7, 0.7, 1.]
        self.axis_color = [0.8, 0.8, 0.8, 1.]
        self.background_color = [0., 0., 0., 1.]

class OWPlot3D(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers), parent)

        self.commands = []
        self.minx = self.miny = self.minz = 0
        self.maxx = self.maxy = self.maxz = 0
        self.view_cube_edge = 10
        self.camera_distance = 30

        self.yaw = self.pitch = -pi / 4.
        self.rotation_factor = 0.3
        self.update_camera()

        self.ortho_scale = 100.
        self.ortho_near = -1
        self.ortho_far = 2000
        self.perspective_near = 0.1
        self.perspective_far = 2000
        self.camera_fov = 30.
        self.zoom_factor = 2000.
        self.move_factor = 100.

        self.x_axis_title = ''
        self.y_axis_title = ''
        self.z_axis_title = ''
        self.show_x_axis_title = self.show_y_axis_title = self.show_z_axis_title = True

        self.vertex_buffers = []
        self.index_buffers = []
        self.vaos = []

        self.use_ortho = False
        self.show_legend = True
        self.legend = Legend(self)

        self.use_2d_symbols = False
        self.symbol_scale = 1.
        self.transparency = 255
        self.show_grid = True
        self.scale = numpy.array([1., 1., 1.])
        self.additional_scale = [0, 0, 0]
        self.scale_x_axis = True
        self.scale_factor = 0.05
        self.data_scale = numpy.array([1., 1., 1.])
        self.data_center = numpy.array([0., 0., 0.])
        self.zoomed_size = [self.view_cube_edge,
                            self.view_cube_edge,
                            self.view_cube_edge]

        self.state = PlotState.IDLE

        self.build_axes()
        self.selections = []
        self.selection_changed_callback = None
        self.selection_updated_callback = None
        self.selection_type = SelectionType.ZOOM
        self.new_selection = None

        self.setMouseTracking(True)
        self.mouseover_callback = None

        self.x_axis_map = None
        self.y_axis_map = None
        self.z_axis_map = None

        self.zoom_stack = []
        self.translation = numpy.array([0., 0., 0.])

        self._theme = LightTheme()
        self.show_axes = True
        self.show_chassis = True

        self.tooltip_fbo_dirty = True
        self.selection_fbo_dirty = True
        self.use_fbos = True

        self.draw_point_cloud = False
        self.hide_outside = False

    def __del__(self):
        # TODO: check if anything needs deleting
        pass

    def initializeGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_CULL_FACE)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

        self.symbol_shader = QtOpenGL.QGLShaderProgram()
        vertex_shader_source = '''
            #extension GL_EXT_gpu_shader4 : enable

            attribute vec4 position;
            attribute vec3 offset;
            attribute vec4 color;
            attribute vec3 normal;

            uniform bool use_2d_symbols;
            uniform bool shrink_symbols;
            uniform bool encode_color;
            uniform bool hide_outside;
            uniform vec4 force_color;
            uniform vec2 transparency; // vec2 instead of float, fixing a bug on windows
                                       // (setUniformValue with float crashes)
			uniform vec2 symbol_scale;
            uniform vec2 view_edge;

            uniform vec3 scale;
            uniform vec3 translation;

            varying vec4 var_color;

            void main(void) {
              vec3 offset_rotated = offset;
              offset_rotated.x *= symbol_scale.x;
              offset_rotated.y *= symbol_scale.x;
              offset_rotated.z *= symbol_scale.x;

              if (use_2d_symbols) {
                  // Calculate inverse of rotations (in this case, inverse
                  // is actually just transpose), so that polygons face
                  // camera all the time.
                  mat3 invs;

                  invs[0][0] = gl_ModelViewMatrix[0][0];
                  invs[0][1] = gl_ModelViewMatrix[1][0];
                  invs[0][2] = gl_ModelViewMatrix[2][0];

                  invs[1][0] = gl_ModelViewMatrix[0][1];
                  invs[1][1] = gl_ModelViewMatrix[1][1];
                  invs[1][2] = gl_ModelViewMatrix[2][1];

                  invs[2][0] = gl_ModelViewMatrix[0][2];
                  invs[2][1] = gl_ModelViewMatrix[1][2];
                  invs[2][2] = gl_ModelViewMatrix[2][2];

                  offset_rotated = invs * offset_rotated;
              }

              vec3 pos = position.xyz;
              pos += translation;
              pos *= scale;
              vec4 off_pos = vec4(pos, 1.);

              if (shrink_symbols) {
                  // Shrink symbols into points by ignoring offsets.
                  gl_PointSize = 2.;
              }
              else {
                  off_pos = vec4(pos+offset_rotated, 1.);
              }

              gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * off_pos;

              if (force_color.a > 0.) {
                var_color = force_color;
              }
              else if (encode_color) {
                // We've packed example index into .w component of this vertex,
                // to output it to the screen, it has to be broken down into RGBA.
                uint index = uint(position.w);
                var_color = vec4(float((index & 0xFF)) / 255.,
                                 float((index & 0xFF00) >> 8) / 255.,
                                 float((index & 0xFF0000) >> 16) / 255.,
                                 float((index & 0xFF000000) >> 24) / 255.);
              }
              else {
                pos = abs(pos);
                float manhattan_distance = max(max(pos.x, pos.y), pos.z)+5.;
                float a = min(pow(min(1., view_edge.x / manhattan_distance), 5.), transparency.x);
                if (use_2d_symbols) {
                    var_color = vec4(color.rgb, a);
                }
                else {
                    // Calculate the amount of lighting this triangle receives (diffuse component only).
                    // The calculations are physically wrong, but look better. TODO: make them look better
                    vec3 light_direction = normalize(vec3(1., 1., 0.5));
                    float diffuse = max(0.,
                        dot(normalize((gl_ModelViewMatrix * vec4(normal, 0.)).xyz), light_direction));
                    var_color = vec4(color.rgb+diffuse*0.7, a);
                }
                if (manhattan_distance > view_edge.x && hide_outside)
                    var_color.a = 0.;
              }
            }
            '''

        fragment_shader_source = '''
            varying vec4 var_color;

            void main(void) {
              gl_FragColor = var_color;
            }
            '''

        self.symbol_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self.symbol_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self.symbol_shader.bindAttributeLocation('position', 0)
        self.symbol_shader.bindAttributeLocation('offset',   1)
        self.symbol_shader.bindAttributeLocation('color',    2)
        self.symbol_shader.bindAttributeLocation('normal',   3)

        if not self.symbol_shader.link():
            print('Failed to link symbol shader!')
        else:
            print('Symbol shader linked.')
        self.symbol_shader_use_2d_symbols = self.symbol_shader.uniformLocation('use_2d_symbols')
        self.symbol_shader_symbol_scale   = self.symbol_shader.uniformLocation('symbol_scale')
        self.symbol_shader_transparency   = self.symbol_shader.uniformLocation('transparency')
        self.symbol_shader_view_edge      = self.symbol_shader.uniformLocation('view_edge')
        self.symbol_shader_scale          = self.symbol_shader.uniformLocation('scale')
        self.symbol_shader_translation    = self.symbol_shader.uniformLocation('translation')
        self.symbol_shader_shrink_symbols = self.symbol_shader.uniformLocation('shrink_symbols')
        self.symbol_shader_encode_color   = self.symbol_shader.uniformLocation('encode_color')
        self.symbol_shader_hide_outside   = self.symbol_shader.uniformLocation('hide_outside')
        self.symbol_shader_force_color    = self.symbol_shader.uniformLocation('force_color')

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
        self.tooltip_fbo = QtOpenGL.QGLFramebufferObject(1024, 1024, format)
        if self.tooltip_fbo.isValid():
            print('Tooltip FBO created.')
        else:
            print('Failed to create tooltip FBO! Tooltips disabled.')
            self.use_fbos = False

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    def update_camera(self):
        self.pitch = clamp(self.pitch, -3., -0.1)
        self.camera = [
            sin(self.pitch)*cos(self.yaw),
            cos(self.pitch),
            sin(self.pitch)*sin(self.yaw)]

    def paintGL(self):
        glClearColor(*self._theme.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if len(self.commands) == 0:
            return

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = self.width(), self.height()
        if self.use_ortho:
            glOrtho(-width / self.ortho_scale,
                     width / self.ortho_scale,
                    -height / self.ortho_scale,
                     height / self.ortho_scale,
                     self.ortho_near,
                     self.ortho_far)
        else:
            aspect = float(width) / height if height != 0 else 1
            gluPerspective(self.camera_fov, aspect, self.perspective_near, self.perspective_far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.camera[0]*self.camera_distance,
            self.camera[1]*self.camera_distance,
            self.camera[2]*self.camera_distance,
            0,-1, 0,
            0, 1, 0)

        if self.show_chassis:
            self.draw_chassis()
        self.draw_grid_and_axes()

        for (cmd, params) in self.commands:
            if cmd == 'scatter':
                vao_id, (X, Y, Z), labels = params
                scale = numpy.maximum([0., 0., 0.], self.scale + self.additional_scale)

                self.symbol_shader.bind()
                self.symbol_shader.setUniformValue(self.symbol_shader_use_2d_symbols, self.use_2d_symbols)
                self.symbol_shader.setUniformValue(self.symbol_shader_shrink_symbols, self.draw_point_cloud)
                self.symbol_shader.setUniformValue(self.symbol_shader_encode_color,   False)
                self.symbol_shader.setUniformValue(self.symbol_shader_hide_outside,   self.hide_outside)
				# Specifying float uniforms with vec2 because of a weird bug in PyQt
                self.symbol_shader.setUniformValue(self.symbol_shader_view_edge,      self.view_cube_edge, self.view_cube_edge)
                self.symbol_shader.setUniformValue(self.symbol_shader_symbol_scale,   self.symbol_scale, self.symbol_scale)
                self.symbol_shader.setUniformValue(self.symbol_shader_transparency,   self.transparency / 255., self.transparency / 255.)
                self.symbol_shader.setUniformValue(self.symbol_shader_scale,          *scale)
                self.symbol_shader.setUniformValue(self.symbol_shader_translation,    *self.translation)
                self.symbol_shader.setUniformValue(self.symbol_shader_force_color,    0., 0., 0., 0.)

                glBindVertexArray(vao_id)
                if self.draw_point_cloud:
                    glDisable(GL_DEPTH_TEST)
                    glDisable(GL_BLEND)
                    glDrawArrays(GL_POINTS, 0, vao_id.num_3d_vertices)
                else:
                    glEnable(GL_DEPTH_TEST)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    if self.use_2d_symbols:
                        glDrawArrays(GL_TRIANGLES, vao_id.num_3d_vertices, vao_id.num_2d_vertices)
                        # Draw outlines (somewhat dark, discuss)
                        #self.symbol_shader.setUniformValue(self.symbol_shader_force_color,
                            #0., 0., 0., self.transparency / 255. + 0.01)
                        #glDisable(GL_DEPTH_TEST)
                        #glDrawArrays(GL_LINES, vao_id.num_3d_vertices+vao_id.num_2d_vertices, vao_id.num_edge_vertices)
                        #self.symbol_shader.setUniformValue(self.symbol_shader_force_color, 0., 0., 0., 0.)
                    else:
                        glDrawArrays(GL_TRIANGLES, 0, vao_id.num_3d_vertices)
                glBindVertexArray(0)

                self.symbol_shader.release()

                if labels != None:
                    glColor4f(*self._theme.labels_color)
                    for x, y, z, label in zip(X, Y, Z, labels):
                        x, y, z = self.transform_data_to_plot((x, y, z))
                        if isinstance(label, str):
                            self.renderText(x,y,z, label, font=self._theme.labels_font)
                        else:
                            self.renderText(x,y,z, ('%f' % label).rstrip('0').rstrip('.'),
                                            font=self._theme.labels_font)
            elif cmd == 'custom':
                callback = params
                callback()

        if self.selection_fbo_dirty:
            self.selection_fbo.bind()
            glClearColor(1, 1, 1, 1)
            glClearStencil(0)
            glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
            self.selection_fbo.release()

        if self.tooltip_fbo_dirty:
            self.tooltip_fbo.bind()
            glClearColor(1, 1, 1, 1)
            glClearDepth(1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.tooltip_fbo.release()

        for (cmd, params) in self.commands:
            if cmd == 'scatter':
                # Don't draw auxiliary info when rotating or selecting
                # (but make sure these are drawn the very first frame
                # plot returns to state idle because the user might want to
                # get tooltips or has just selected a bunch of stuff).
                vao_id, _, _ = params

                if self.tooltip_fbo_dirty:
                    # Draw data the same as to the screen, but with
                    # disabled blending and enabled depth testing.
                    self.tooltip_fbo.bind()
                    glDisable(GL_BLEND)
                    glEnable(GL_DEPTH_TEST)

                    self.symbol_shader.bind()
                    # Most uniforms retain their values.
                    self.symbol_shader.setUniformValue(self.symbol_shader_encode_color, True)
                    self.symbol_shader.setUniformValue(self.symbol_shader_shrink_symbols, False)
                    glBindVertexArray(vao_id)
                    glDrawArrays(GL_TRIANGLES, 0, vao_id.num_3d_vertices)
                    glBindVertexArray(0)
                    self.symbol_shader.release()
                    self.tooltip_fbo.release()
                    self.tooltip_fbo_dirty = False

                if self.selection_fbo_dirty:
                    # Draw data as points instead, this means that examples farther away
                    # will still have a good chance at being visible (not covered).
                    self.selection_fbo.bind()
                    self.symbol_shader.bind()
                    self.symbol_shader.setUniformValue(self.symbol_shader_encode_color, True)
                    self.symbol_shader.setUniformValue(self.symbol_shader_shrink_symbols, True)
                    glDisable(GL_DEPTH_TEST)
                    glDisable(GL_BLEND)
                    glBindVertexArray(vao_id)
                    glDrawArrays(GL_POINTS, 0, vao_id.num_3d_vertices)
                    glBindVertexArray(0)
                    self.symbol_shader.release()

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

    def draw_helpers(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if self.state == PlotState.SCALING:
            x, y = self.mouse_pos.x(), self.mouse_pos.y()
            glColor4f(*self._theme.helpers_color)
            draw_triangle(x-5, y-30, x+5, y-30, x, y-40)
            draw_line(x, y, x, y-30)
            draw_triangle(x-5, y-10, x+5, y-10, x, y)

            draw_triangle(x+10, y, x+20, y-5, x+20, y+5)
            draw_line(x+10, y, x+40, y)
            draw_triangle(x+50, y, x+40, y-5, x+40, y+5)

            self.renderText(x, y-50, 'Scale y axis', font=self._theme.labels_font)
            self.renderText(x+60, y+3,
                            'Scale {0} axis'.format(['z', 'x'][self.scale_x_axis]),
                            font=self._theme.labels_font)
        elif self.state == PlotState.SELECTING and self.new_selection != None:
            self.new_selection.draw()

        for selection in self.selections:
            selection.draw()

    def set_x_axis_title(self, title):
        self.x_axis_title = title
        self.updateGL()

    def set_show_x_axis_title(self, show):
        self.show_x_axis_title = show
        self.updateGL()

    def set_y_axis_title(self, title):
        self.y_axis_title = title
        self.updateGL()

    def set_show_y_axis_title(self, show):
        self.show_y_axis_title = show
        self.updateGL()

    def set_z_axis_title(self, title):
        self.z_axis_title = title
        self.updateGL()

    def set_show_z_axis_title(self, show):
        self.show_z_axis_title = show
        self.updateGL()

    def draw_chassis(self):
        glColor4f(*self._theme.axis_values_color)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, 0x00FF)
        edges = [self.x_axis, self.y_axis, self.z_axis,
                 self.x_axis+self.unit_z, self.x_axis+self.unit_y,
                 self.x_axis+self.unit_z+self.unit_y,
                 self.y_axis+self.unit_x, self.y_axis+self.unit_z,
                 self.y_axis+self.unit_x+self.unit_z,
                 self.z_axis+self.unit_x, self.z_axis+self.unit_y,
                 self.z_axis+self.unit_x+self.unit_y]
        glBegin(GL_LINES)
        for edge in edges:
            start, end = edge
            glVertex3f(*start)
            glVertex3f(*end)
        glEnd()
        glDisable(GL_LINE_STIPPLE)

    def draw_grid_and_axes(self):
        cam_in_space = numpy.array([
          self.camera[0]*self.camera_distance,
          self.camera[1]*self.camera_distance,
          self.camera[2]*self.camera_distance
        ])

        def plane_visible(plane):
            normal = normal_from_points(*plane[:3])
            cam_plane = normalize(plane[0] - cam_in_space)
            if numpy.dot(normal, cam_plane) > 0:
                return False
            return True

        def draw_axis(line):
            glColor4f(*self._theme.axis_color)
            glLineWidth(2)
            glBegin(GL_LINES)
            glVertex3f(*line[0])
            glVertex3f(*line[1])
            glEnd()

        def draw_discrete_axis_values(axis, coord_index, normal, axis_map):
            start, end = axis
            start_value = self.transform_plot_to_data(numpy.copy(start))[coord_index]
            end_value = self.transform_plot_to_data(numpy.copy(end))[coord_index]
            length = end_value - start_value
            offset = normal*0.8
            for key in axis_map.keys():
                if start_value <= key <= end_value:
                    position = start + (end-start)*((key-start_value) / length)
                    glBegin(GL_LINES)
                    glVertex3f(*(position))
                    glVertex3f(*(position+normal*0.2))
                    glEnd()
                    position += offset
                    self.renderText(position[0],
                                    position[1],
                                    position[2],
                                    axis_map[key], font=self._theme.labels_font)

        def draw_values(axis, coord_index, normal, axis_map):
            glColor4f(*self._theme.axis_values_color)
            glLineWidth(1)
            if axis_map != None:
                draw_discrete_axis_values(axis, coord_index, normal, axis_map)
                return
            start, end = axis
            start_value = self.transform_plot_to_data(numpy.copy(start))[coord_index]
            end_value = self.transform_plot_to_data(numpy.copy(end))[coord_index]
            values, num_frac = loose_label(start_value, end_value, 7)
            format = '%%.%df' % num_frac
            offset = normal*0.8
            for value in values:
                if not (start_value <= value <= end_value):
                    continue
                position = start + (end-start)*((value-start_value) / float(end_value-start_value))
                glBegin(GL_LINES)
                glVertex3f(*(position))
                glVertex3f(*(position+normal*0.2))
                glEnd()
                value = self.transform_plot_to_data(numpy.copy(position))[coord_index]
                position += offset
                self.renderText(position[0],
                                position[1],
                                position[2],
                                format % value)

        def draw_axis_title(axis, title, normal):
            middle = (axis[0] + axis[1]) / 2.
            middle += normal * 1. if axis[0][1] != axis[1][1] else normal * 2.
            self.renderText(middle[0], middle[1], middle[2],
                            title,
                            font=self._theme.axis_title_font)

        def draw_grid(axis0, axis1, normal0, normal1, i, j):
            glColor4f(*self._theme.grid_color)
            for axis, normal, coord_index in zip([axis0, axis1], [normal0, normal1], [i, j]):
                start, end = axis
                start_value = self.transform_plot_to_data(numpy.copy(start))[coord_index]
                end_value = self.transform_plot_to_data(numpy.copy(end))[coord_index]
                values, _ = loose_label(start_value, end_value, 7)
                for value in values:
                    if not (start_value <= value <= end_value):
                        continue
                    position = start + (end-start)*((value-start_value) / float(end_value-start_value))
                    glBegin(GL_LINES)
                    glVertex3f(*position)
                    glVertex3f(*(position-normal*10.))
                    glEnd()

        glDisable(GL_DEPTH_TEST)
        glLineWidth(1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        planes = [self.axis_plane_xy, self.axis_plane_yz,
                  self.axis_plane_xy_back, self.axis_plane_yz_right]
        axes = [[self.x_axis, self.y_axis],
                [self.y_axis, self.z_axis],
                [self.x_axis+self.unit_z, self.y_axis+self.unit_z],
                [self.z_axis+self.unit_x, self.y_axis+self.unit_x]]
        normals = [[numpy.array([0,-1, 0]), numpy.array([-1, 0, 0])],
                   [numpy.array([0, 0,-1]), numpy.array([ 0,-1, 0])],
                   [numpy.array([0,-1, 0]), numpy.array([-1, 0, 0])],
                   [numpy.array([0,-1, 0]), numpy.array([ 0, 0,-1])]]
        coords = [[0, 1],
                  [1, 2],
                  [0, 1],
                  [2, 1]]
        visible_planes = map(plane_visible, planes)
        xz_visible = not plane_visible(self.axis_plane_xz)
        if self.show_grid:
            if xz_visible:
                draw_grid(self.x_axis, self.z_axis, numpy.array([0,0,-1]), numpy.array([-1,0,0]), 0, 2)
            for visible, (axis0, axis1), (normal0, normal1), (i, j) in\
                 zip(visible_planes, axes, normals, coords):
                if not visible:
                    draw_grid(axis0, axis1, normal0, normal1, i, j)

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

        if not self.show_axes:
            return

        if visible_planes[0 if xz_visible else 2]:
            draw_axis(self.x_axis)
            draw_values(self.x_axis, 0, numpy.array([0, 0, -1]), self.x_axis_map)
            if self.show_x_axis_title:
                draw_axis_title(self.x_axis, self.x_axis_title, numpy.array([0, 0, -1]))
        elif visible_planes[2 if xz_visible else 0]:
            draw_axis(self.x_axis + self.unit_z)
            draw_values(self.x_axis + self.unit_z, 0, numpy.array([0, 0, 1]), self.x_axis_map)
            if self.show_x_axis_title:
                draw_axis_title(self.x_axis + self.unit_z,
                                self.x_axis_title, numpy.array([0, 0, 1]))

        if visible_planes[1 if xz_visible else 3]:
            draw_axis(self.z_axis)
            draw_values(self.z_axis, 2, numpy.array([-1, 0, 0]), self.z_axis_map)
            if self.show_z_axis_title:
                draw_axis_title(self.z_axis, self.z_axis_title, numpy.array([-1, 0, 0]))
        elif visible_planes[3 if xz_visible else 1]:
            draw_axis(self.z_axis + self.unit_x)
            draw_values(self.z_axis + self.unit_x, 2, numpy.array([1, 0, 0]), self.z_axis_map)
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
        draw_values(axis, 1, normal, self.y_axis_map)
        if self.show_y_axis_title:
            draw_axis_title(axis, self.y_axis_title, normal)

        # Remember which axis to scale when dragging mouse horizontally.
        self.scale_x_axis = False if rightmost_visible % 2 == 0 else True

    def build_axes(self):
        edge_half = self.view_cube_edge / 2.
        x_axis = [[-edge_half,-edge_half,-edge_half], [edge_half,-edge_half,-edge_half]]
        y_axis = [[-edge_half,-edge_half,-edge_half], [-edge_half,edge_half,-edge_half]]
        z_axis = [[-edge_half,-edge_half,-edge_half], [-edge_half,-edge_half,edge_half]]

        self.x_axis = x_axis = numpy.array(x_axis)
        self.y_axis = y_axis = numpy.array(y_axis)
        self.z_axis = z_axis = numpy.array(z_axis)

        self.unit_x = unit_x = numpy.array([self.view_cube_edge,0,0])
        self.unit_y = unit_y = numpy.array([0,self.view_cube_edge,0])
        self.unit_z = unit_z = numpy.array([0,0,self.view_cube_edge])
 
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

    def scatter(self, X, Y, Z, colors='b', sizes=5, symbols=None, labels=None, **kwargs):
        if len(X) != len(Y) != len(Z):
            raise ValueError('Axis data arrays must be of equal length')
        num_points = len(X)

        if isinstance(colors, str):
            color_map = {'r': [1.0, 0.0, 0.0, 1.0],
                         'g': [0.0, 1.0, 0.0, 1.0],
                         'b': [0.0, 0.0, 1.0, 1.0]}
            default = [0.0, 0.0, 1.0, 1.0]
            colors = [color_map.get(colors, default) for _ in range(num_points)]
 
        if isinstance(sizes, (int, float)):
            sizes = [sizes for _ in range(num_points)]

        # Scale sizes to 0..1
        self.max_size = float(numpy.max(sizes))
        sizes = [size / self.max_size for size in sizes]

        if symbols == None:
            symbols = [Symbol.TRIANGLE for _ in range(num_points)]

        # We scale and translate data into almost-unit cube centered around (0,0,0) in plot-space.
        # It's almost-unit because the length of its edge is specified with view_cube_edge.
        # This transform is done to ease later calculations and for presentation purposes.
        min = self.min_x, self.min_y, self.min_z = numpy.min(X), numpy.min(Y), numpy.min(Z)
        max = self.max_x, self.max_y, self.max_z = numpy.max(X), numpy.max(Y), numpy.max(Z)
        min = numpy.array(min)
        max = numpy.array(max)
        range_x, range_y, range_z = max-min
        self.data_center = (min + max) / 2 

        scale_x = self.view_cube_edge / range_x
        scale_y = self.view_cube_edge / range_y
        scale_z = self.view_cube_edge / range_z

        self.data_scale = numpy.array([scale_x, scale_y, scale_z])

        # TODO: if self.use_2d_symbols

        num_3d_vertices = 0
        num_2d_vertices = 0
        num_edge_vertices = 0
        vertices = []
        ai = -1 # Array index (used in color-picking).
        for x, y, z, (r,g,b,a), size, symbol in zip(X, Y, Z, colors, sizes, symbols):
            x -= self.data_center[0]
            y -= self.data_center[1]
            z -= self.data_center[2]
            x *= scale_x
            y *= scale_y
            z *= scale_z
            triangles = get_symbol_data(symbol)
            ss = size*0.02
            ai += 1
            for v0, v1, v2, n0, n1, n2 in triangles:
                num_3d_vertices += 3
                vertices.extend([x,y,z, ai, ss*v0[0],ss*v0[1],ss*v0[2], r,g,b,a, n0[0],n0[1],n0[2],
                                 x,y,z, ai, ss*v1[0],ss*v1[1],ss*v1[2], r,g,b,a, n1[0],n1[1],n1[2],
                                 x,y,z, ai, ss*v2[0],ss*v2[1],ss*v2[2], r,g,b,a, n2[0],n2[1],n2[2]])

        for x, y, z, (r,g,b,a), size, symbol in zip(X, Y, Z, colors, sizes, symbols):
            x -= self.data_center[0]
            y -= self.data_center[1]
            z -= self.data_center[2]
            x *= scale_x
            y *= scale_y
            z *= scale_z
            triangles = get_2d_symbol_data(symbol)
            ss = size*0.02
            for v0, v1, v2, _, _, _ in triangles:
                num_2d_vertices += 3
                vertices.extend([x,y,z, 0, ss*v0[0],ss*v0[1],ss*v0[2], r,g,b,a, 0,0,0,
                                 x,y,z, 0, ss*v1[0],ss*v1[1],ss*v1[2], r,g,b,a, 0,0,0,
                                 x,y,z, 0, ss*v2[0],ss*v2[1],ss*v2[2], r,g,b,a, 0,0,0])

        for x, y, z, (r,g,b,a), size, symbol in zip(X, Y, Z, colors, sizes, symbols):
            x -= self.data_center[0]
            y -= self.data_center[1]
            z -= self.data_center[2]
            x *= scale_x
            y *= scale_y
            z *= scale_z
            edges = get_2d_symbol_edges(symbol)
            ss = size*0.02
            for v0, v1 in edges:
                num_edge_vertices += 2
                vertices.extend([x,y,z, 0, ss*v0[0],ss*v0[1],ss*v0[2], r,g,b,a, 0,0,0,
                                 x,y,z, 0, ss*v1[0],ss*v1[1],ss*v1[2], r,g,b,a, 0,0,0])

        # Build Vertex Buffer + Vertex Array Object.
        vao_id = GLuint(0)
        glGenVertexArrays(1, vao_id)
        glBindVertexArray(vao_id)

        vertex_buffer_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, numpy.array(vertices, 'f'), GL_STATIC_DRAW)

        vertex_size = (4+3+3+4)*4
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(0))    # position
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(4*4))  # offset
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(7*4))  # color
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(11*4)) # normal
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)
        glEnableVertexAttribArray(3)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        vao_id.num_3d_vertices = num_3d_vertices
        vao_id.num_2d_vertices = num_2d_vertices
        vao_id.num_edge_vertices = num_edge_vertices
        self.vertex_buffers.append(vertex_buffer_id)
        self.vaos.append(vao_id)
        self.commands.append(("scatter", [vao_id, (X,Y,Z), labels]))
        self.updateGL()

    def set_x_axis_map(self, map):
        self.x_axis_map = map
        self.updateGL()

    def set_y_axis_map(self, map):
        self.y_axis_map = map
        self.updateGL()

    def set_z_axis_map(self, map):
        self.z_axis_map = map
        self.updateGL()

    def set_new_zoom(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.selections = []
        self.zoom_stack.append((self.scale, self.translation))

        max = numpy.array([x_max, y_max, z_max])
        min = numpy.array([x_min, y_min, z_min])
        min, max = map(numpy.copy, [min, max])
        min -= self.data_center
        min *= self.data_scale
        max -= self.data_center
        max *= self.data_scale
        center = (max + min) / 2.
        new_translation = -numpy.array(center)
        # Avoid division by zero by adding a small value (this happens when zooming in
        # on elements with the same value of an attribute).
        self.zoomed_size = numpy.array(map(lambda i: i+0.001 if i == 0 else i, max-min))
        new_scale = self.view_cube_edge / self.zoomed_size
        self._animate_new_scale_translation(new_scale, new_translation)

    def _animate_new_scale_translation(self, new_scale, new_translation, num_steps=10):
        translation_step = (new_translation - self.translation) / float(num_steps)
        scale_step = (new_scale - self.scale) / float(num_steps)
        # Animate zooming: translate first for a number of steps,
        # then scale. Make sure it doesn't take too long.
        start = time.time()
        for i in range(num_steps):
            if time.time() - start > 1.:
                self.translation = new_translation
                break
            self.translation = self.translation + translation_step
            self.updateGL()
        for i in range(num_steps):
            if time.time() - start > 1.:
                self.scale = new_scale
                break
            self.scale = self.scale + scale_step
            self.updateGL()

    def pop_zoom(self):
        if len(self.zoom_stack) < 1:
            new_translation = numpy.array([0., 0., 0.])
            new_scale = numpy.array([1., 1., 1.])
        else:
            new_scale, new_translation = self.zoom_stack.pop()
        self._animate_new_scale_translation(new_scale, new_translation)
        self.zoomed_size = self.view_cube_edge / new_scale

    def save_to_file(self):
        size_dlg = OWChooseImageSizeDlg(self, [], parent=self)
        size_dlg.exec_()

    def save_to_file_direct(self, file_name, size=None):
        img = self.grabFrameBuffer()
        if size != None:
            img = img.scaled(size)
        return img.save(file_name)

    def transform_data_to_plot(self, vertex):
        vertex -= self.data_center
        vertex *= self.data_scale
        vertex += self.translation
        vertex *= numpy.maximum([0., 0., 0.], self.scale + self.additional_scale)
        return vertex

    def transform_plot_to_data(self, vertex):
        denominator = numpy.maximum([0., 0., 0.], self.scale + self.additional_scale)
        denominator = numpy.array(map(lambda v: v+0.00001 if v == 0. else v, denominator))
        vertex /= denominator
        vertex -= self.translation
        vertex /= self.data_scale
        vertex += self.data_center
        return vertex

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
            projection = QMatrix4x4()
            if self.use_ortho:
                projection.ortho(-width / self.ortho_scale, width / self.ortho_scale,
                                 -height / self.ortho_scale, height / self.ortho_scale,
                                 self.ortho_near, self.ortho_far)
            else:
                projection.perspective(self.camera_fov, float(width) / height,
                                       self.perspective_near, self.perspective_far)

            modelview = QMatrix4x4()
            modelview.lookAt(QVector3D(self.camera[0]*self.camera_distance,
                                       self.camera[1]*self.camera_distance,
                                       self.camera[2]*self.camera_distance),
                             QVector3D(0,-1, 0),
                             QVector3D(0, 1, 0))

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
            for (cmd, params) in self.commands:
                if cmd == 'scatter':
                    _, (X, Y, Z), _ = params
                    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
                        x, y, z = self.transform_data_to_plot((x,y,z))
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
                self.additional_scale = [0., 0., 0.]
            else:
                self.pop_zoom()
            self.updateGL()
        elif buttons & Qt.MiddleButton:
            self.state = PlotState.ROTATING
            self.selections = []
            self.new_selection = None

    def mouseMoveEvent(self, event):
        pos = event.pos()

        if self.mouseover_callback != None and self.state == PlotState.IDLE and\
            (not self.show_legend or not self.legend.contains(pos.x(), pos.y())):
            # Use pixel-color-picking to read example index under mouse cursor.
            self.tooltip_fbo.bind()
            value = glReadPixels(pos.x(), self.height() - pos.y(),
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
                right_scale = self.width()*max(self.scale[0], self.scale[2])*0.1
                up_scale = self.height()*self.scale[1]*0.1
                self.translation -= right_vec*(dx / right_scale) +\
                                    up_vec*(dy / up_scale)
            else:
                self.yaw += dx / (self.rotation_factor*self.width())
                self.pitch += dy / (self.rotation_factor*self.height())
                self.update_camera()
        elif self.state == PlotState.SCALING:
            dx = pos.x() - self.scaling_init_pos.x()
            dy = pos.y() - self.scaling_init_pos.y()
            dx /= float(self.zoomed_size[0 if self.scale_x_axis else 2])
            dy /= float(self.zoomed_size[1])
            dx /= self.scale_factor * self.width()
            dy /= self.scale_factor * self.height()
            self.additional_scale = [dx, dy, 0] if self.scale_x_axis else [0, dy, dx]
        elif self.state == PlotState.PANNING:
            self.dragged_selection.move(dx, dy)

        self.mouse_pos = pos
        self.updateGL()

    def mouseReleaseEvent(self, event):
        if self.state == PlotState.SELECTING and self.new_selection == None:
            self.new_selection = PolygonSelection(self, [event.pos().x(), event.pos().y()])
            return

        if self.state == PlotState.SCALING:
            self.scale = numpy.maximum([0., 0., 0.], self.scale + self.additional_scale)
            self.additional_scale = [0., 0., 0.]
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
            self.scale *= delta
            self.tooltip_fbo_dirty = True
            self.updateGL()

    def remove_last_selection(self):
        if len(self.selections) > 0:
            self.selections.pop()
            self.updateGL()
            self.selection_changed_callback() if self.selection_changed_callback else None

    def remove_all_selections(self):
        self.selections = []
        self.selection_changed_callback() if self.selection_changed_callback else None
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
        self.commands = []
        self.selections = []
        self.legend.clear()
        self.zoom_stack = []
        self.zoomed_size = [self.view_cube_edge,
                            self.view_cube_edge,
                            self.view_cube_edge]
        self.translation = numpy.array([0., 0., 0.])
        self.scale = numpy.array([1., 1., 1.])
        self.additional_scale = numpy.array([0., 0., 0.])
        self.x_axis_title = self.y_axis_title = self.z_axis_title = ''
        self.x_axis_map = self.y_axis_map = self.z_axis_map = None
        self.tooltip_fbo_dirty = True
        self.selection_fbo_dirty = True
        self.updateGL()


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
