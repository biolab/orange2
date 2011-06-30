"""
"""

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtOpenGL

import orange

from OpenGL.GL import *
from OpenGL.GLU import *
from ctypes import byref, c_char_p, c_int, create_string_buffer
import sys
import numpy
from math import sin, cos

# Import undefined functions.
try:
  from OpenGL import platform
  gl = platform.OpenGL
except ImportError:
  try:
    gl = cdll.LoadLibrary('libGL.so')
  except OSError:
    from ctypes.util import find_library
    path = find_library('OpenGL')
    gl = cdll.LoadLibrary(path)

glCreateProgram = gl.glCreateProgram
glCreateShader = gl.glCreateShader
glShaderSource = gl.glShaderSource
glCompileShader = gl.glCompileShader
glGetShaderiv = gl.glGetShaderiv
glDeleteShader = gl.glDeleteShader


def normalize(vec):
  return vec / numpy.sqrt(numpy.sum(vec** 2))


class OWGraph3D(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers), parent)
        self.commands = []
        self.minx = self.miny = self.minz = 0
        self.maxx = self.maxy = self.maxz = 0
        self.b_box = [numpy.array([0,   0,   0]), numpy.array([0, 0, 0])]
        self.camera = numpy.array([0.6, 0.8, 0]) # Location on a unit sphere around the center. This is where camera is looking from.
        self.center = numpy.array([0,   0,   0])

        # TODO: move to center shortcut (maybe a GUI element?)

        self.yaw = self.pitch = 0.
        self.rotation_factor = 100.
        self.zoom_factor = 100.
        self.zoom = 10.
        self.move_factor = 100.
        self.mouse_pos = [100, 100] # TODO: get real mouse position, calculate camera, fix the initial jump
        self.update_axes()

        self.axis_title_font = QFont('Helvetica', 10, QFont.Bold)
        self.ticks_font = QFont('Helvetica', 9)
        self.x_axis_title = ''
        self.y_axis_title = ''
        self.z_axis_title = ''

        self.color_shader = glCreateProgram()
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)

        vertex_shader_source = '''
            attribute vec4 position;
            attribute vec4 color;

            uniform mat4 projection;
            uniform mat4 modelview;

            varying vec4 var_color;

            void main(void) {
              gl_Position = projection * modelview * position;
              var_color = color;
            }
            '''

        fragment_shader_source = '''
            varying vec4 var_color;

            void main(void) {
              gl_FragColor = var_color;
            }
            '''

        vertex_shader_source = c_char_p(vertex_shader_source)
        fragment_shader_source = c_char_p(fragment_shader_source)

        def print_log(shader):
          length = c_int()
          glGetShaderiv(shader, GL_INFO_LOG_LENGTH, byref(length))

          if length.value > 0:
            log = create_string_buffer(length.value)
            glGetShaderInfoLog(shader, length, byref(length), log)
            print(log.value)

        length = c_int(-1)
        for shader, source in zip([vertex_shader, fragment_shader], [vertex_shader_source, fragment_shader_source]):
          glShaderSource(shader, 1, byref(source), byref(length));
          glCompileShader(shader)
          status = c_int()
          glGetShaderiv(shader, GL_COMPILE_STATUS, byref(status))
          if not status.value:
            print('Failed to compile shader:')
            print_log(shader)
            glDeleteShader(shader)

        print('Shaders compiled!')


    def initializeGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(30.0, float(self.width())/float(self.height()), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
 

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if height == 0:
            aspect = 1
        else:
            aspect = float(width) / height
        gluPerspective(30.0, aspect, 0.1, 100)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.camera[0]*self.zoom + self.center[0],
            self.camera[1]*self.zoom + self.center[1],
            self.camera[2]*self.zoom + self.center[2],
            self.center[0],
            self.center[1],
            self.center[2],
            0, 1, 0)
        self.paint_axes()

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        for cmd, vertex_buffer in self.commands:
          self.color_shader.bind()
          self.color_shader.setAttributeBuffer(0, GL_FLOAT, 0,   4, 8*4)
          self.color_shader.setAttributeBuffer(1, GL_FLOAT, 4*4, 4, 8*4)
          self.color_shader.enableAttributeArray(0)
          self.color_shader.enableAttributeArray(1)

          glDrawArrays(GL_TRIANGLES, 0, vertex_buffer.num_vertices);

          self.color_shader.disableAttributeArray(0)
          self.color_shader.disableAttributeArray(1)
          glUseProgram(0)

        glDisable(GL_CULL_FACE)

    def set_x_axis_title(self, title):
      self.x_axis_title = title
      self.updateGL()

    def set_y_axis_title(self, title):
      self.y_axis_title = title
      self.updateGL()

    def set_z_axis_title(self, title):
      self.z_axis_title = title
      self.updateGL()

    def paint_axes(self):
        glDisable(GL_CULL_FACE)
        glColor4f(1,1,1,1)
        for start, end in [self.x_axis, self.y_axis, self.z_axis]:
            glBegin(GL_LINES)
            glVertex3f(*start)
            glVertex3f(*end)
            glEnd()

        bb_center = (self.b_box[1] + self.b_box[0]) / 2.

        # Draw axis labels.
        glColor4f(0,0,0,1)

        ac = (self.x_axis[0] + self.x_axis[1]) / 2.
        self.renderText(ac[0], ac[1]-0.2, ac[2]-0.2, self.x_axis_title, font=self.axis_title_font)

        glPushMatrix();
        ac = (self.y_axis[0] + self.y_axis[1]) / 2.
        glTranslatef(ac[0], ac[1]-0.2, ac[2]-0.2)
        glRotatef(90, 1,0,0);
        #self.renderText(ac[0], ac[1]-0.2, ac[2]-0.2, self.YaxisTitle, font=self.axisTitleFont)
        self.renderText(0,0,0, self.y_axis_title, font=self.axis_title_font)
        glPopMatrix();

        ac = (self.z_axis[0] + self.z_axis[1]) / 2.
        self.renderText(ac[0], ac[1]-0.2, ac[2]-0.2, self.z_axis_title, font=self.axis_title_font)

        outwards = normalize(self.x_axis[0] - bb_center)
        pos = self.x_axis[0] + outwards * 0.2
        self.renderText(pos[0], pos[1], pos[2], '{0:.2}'.format(pos[0]))

        glColor4f(1,1,1,1)

        def paint_grid(plane_quad, sub=20):
            P11, P12, P22, P21 = numpy.asarray(plane_quad)
            Dx = numpy.linspace(0.0, 1.0, num=sub)
            P1vecH = P12 - P11
            P2vecH = P22 - P21
            P1vecV = P21 - P11
            P2vecV = P22 - P12
            glBegin(GL_LINES)
            for i, dx in enumerate(Dx):
                start = P11 + P1vecH*dx
                end = P21 + P2vecH*dx
                glVertex3f(*start)
                glVertex3f(*end)

                start = P11 + P1vecV*dx
                end = P12 + P2vecV*dx
                glVertex3f(*start)
                glVertex3f(*end)
            glEnd()

        def paint_quad(plane_quad):
            P11, P12, P21, P22 = numpy.asarray(plane_quad)
            glBegin(GL_QUADS)
            glVertex3f(*P11)
            glVertex3f(*P12)
            glVertex3f(*P21)
            glVertex3f(*P22)
            glEnd()

        color_plane = [0.5, 0.5, 0.5, 0.5]
        color_grid = [0.3, 0.3, 0.3, 1.0]

        def paint_plane(plane_quad):
            #glColor4f(*colorPlane)
            #paintQuad(planeQuad)
            glColor4f(*color_grid)
            paint_grid(plane_quad)

        def normal_from_points(P1, P2, P3):
            V1 = P2 - P1
            V2 = P3 - P1
            return normalize(numpy.cross(V1, V2))
 
        def draw_grid_visible(plane_quad, ccw=False):
            normal = normal_from_points(*plane_quad[:3])
            cam_in_space = numpy.array([
              self.center[0] + self.camera[0]*self.zoom,
              self.center[1] + self.camera[1]*self.zoom,
              self.center[2] + self.camera[2]*self.zoom
            ])
            camera_vector = normalize(plane_quad[0] - cam_in_space)
            cos = numpy.dot(normal, camera_vector) * (-1 if ccw else 1)
            if cos > 0:
                paint_plane(plane_quad)

        draw_grid_visible(self.axis_plane_xy)
        draw_grid_visible(self.axis_plane_yz)
        draw_grid_visible(self.axis_plane_xz)
        draw_grid_visible(self.axis_plane_xy_back)
        draw_grid_visible(self.axis_plane_yz_right)
        draw_grid_visible(self.axis_plane_xz_top)

        glEnable(GL_CULL_FACE)

    def update_axes(self):
        x_axis = [[self.minx, self.miny, self.minz],
                  [self.maxx, self.miny, self.minz]]
        y_axis = [[self.minx, self.miny, self.minz],
                  [self.minx, self.maxy, self.minz]]
        z_axis = [[self.minx, self.miny, self.minz],
                  [self.minx, self.miny, self.maxz]]
        self.x_axis = x_axis = numpy.array(x_axis)
        self.y_axis = y_axis = numpy.array(y_axis)
        self.z_axis = z_axis = numpy.array(z_axis)

        self.unit_x = unit_x = numpy.array([self.maxx - self.minx, 0, 0])
        self.unit_y = unit_y = numpy.array([0, self.maxy - self.miny, 0])
        self.unit_z = unit_z = numpy.array([0, 0, self.maxz - self.minz])
 
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

    def scatter(self, X, Y, Z, c="b", s=20, **kwargs):
        array = [[x, y, z] for x,y,z in zip(X, Y, Z)]
        if isinstance(c, str):
            color_map = {"r": [1.0, 0.0, 0.0, 1.0],
                         "g": [0.0, 1.0, 0.0, 1.0],
                         "b": [0.0, 0.0, 1.0, 1.0]}
            default = [0.0, 0.0, 1.0, 1.0]
            colors = [color_map.get(c, default) for _ in array]
        else:
            colors = c
 
        if isinstance(s, int):
            s = [s for _ in array]

        max, min = numpy.max(array, axis=0), numpy.min(array, axis=0)
        self.b_box = [max, min]
        self.minx, self.miny, self.minz = min
        self.maxx, self.maxy, self.maxz = max
        self.center = (min + max) / 2 
        self.normal_size = numpy.max(self.center - self.b_box[1]) / 100.

        vertex_buffer = QtOpenGL.QGLBuffer(QtOpenGL.QGLBuffer.VertexBuffer)
        if vertex_buffer.create() == False:
          print('Warning: vertex buffers not supported')
          return

        vertex_buffer.bind()
        vertex_buffer.setUsagePattern(QtOpenGL.QGLBuffer.StaticDraw)

        vertices = []
        for (x,y,z), (r,g,b,a), size in zip(array, colors, s):
          vertices.extend([x-size*self.normal_size,y,z, r,g,b,a])
          vertices.extend([x+size*self.normal_size,y,z, r,g,b,a])
          vertices.extend([x,y+size*self.normal_size,z, r,g,b,a])

        vertex_buffer.allocate((GLfloat * len(vertices))(vertices), len(vertices)*4)

        self.commands.append(("scatter", (vertex_buffer)))
        self.update_axes()
        self.updateGL()
 
    def mousePressEvent(self, event):
      self.mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
      if event.buttons() & Qt.MiddleButton:
        pos = event.pos()
        dx = pos.x() - self.mouse_pos.x()
        dy = pos.y() - self.mouse_pos.y()
        if QApplication.keyboardModifiers() & Qt.ShiftModifier:
          off_x = numpy.cross(self.camera, [0,1,0]) * (dx / self.move_factor)
          #off_y = numpy.cross(self.camera, [1,0,0]) * (dy / self.move_factor)
          # TODO: this incidentally works almost fine, but the math is wrong and should be fixed
          self.center += off_x
        else:
          self.yaw += dx /  self.rotation_factor
          self.pitch += dy / self.rotation_factor
          self.camera = [
            sin(self.pitch)*cos(self.yaw),
            cos(self.pitch),
            sin(self.pitch)*sin(self.yaw)]
        self.mouse_pos = pos
        self.updateGL()

    def wheelEvent(self, event):
      if event.orientation() == Qt.Vertical:
        self.zoom -= event.delta() / self.zoom_factor
        if self.zoom < 2:
          self.zoom = 2
        self.updateGL()

    def clear(self):
        self.commands = []

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWGraph3D()
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

#    x = [rand()*2 for i in range(N)]
#    y = [rand()*2 for i in range(N)]
#    z = [-3 + rand() for i in range(N)]
#    colors = "b"
    w.scatter(x, y, z, c=colors)
    app.exec_()
