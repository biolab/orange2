"""
    .. class:: OWGraph3D
        Base class for 3D graphs.

    .. attribute:: show_legend
        Determines whether to display the legend or not.

    .. attribute:: ortho
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

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtOpenGL

import orange

import OpenGL
OpenGL.ERROR_CHECKING = True
OpenGL.ERROR_LOGGING = True
#OpenGL.FULL_LOGGING = True
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import ArrayDatatype
from ctypes import byref, c_char_p, c_int, create_string_buffer
import sys
import numpy
from math import sin, cos, pi

# Import undefined functions, override some wrappers.
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
glDeleteProgram = gl.glDeleteProgram
glGetShaderInfoLog = gl.glGetShaderInfoLog
glGenVertexArrays = gl.glGenVertexArrays
glBindVertexArray = gl.glBindVertexArray
glGenBuffers = gl.glGenBuffers
glDeleteBuffers = gl.glDeleteBuffers
glVertexAttribPointer = gl.glVertexAttribPointer
glEnableVertexAttribArray = gl.glEnableVertexAttribArray
glVertexAttribPointer = gl.glVertexAttribPointer
glEnableVertexAttribArray = gl.glEnableVertexAttribArray
glGetProgramiv = gl.glGetProgramiv
glDrawElements = gl.glDrawElements
glDrawArrays = gl.glDrawArrays
glBindBuffer = gl.glBindBuffer
glBufferData = gl.glBufferData


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

        self.labels_font = QFont('Helvetice', 8)
        self.axis_title_font = QFont('Helvetica', 10, QFont.Bold)
        self.ticks_font = QFont('Helvetica', 9)
        self.x_axis_title = ''
        self.y_axis_title = ''
        self.z_axis_title = ''
        self.show_x_axis_title = self.show_y_axis_title = self.show_z_axis_title = True

        self.color_plane = numpy.array([0.95, 0.95, 0.95, 0.3])
        self.color_grid = numpy.array([0.8, 0.8, 0.8, 1.0])

        self.vertex_buffers = []
        self.index_buffers = []
        self.vaos = []

        self.ortho = False
        self.show_legend = True
        self.legend_border_color = [0.5, 0.5, 0.5, 1]
        self.legend_border_thickness = 2
        self.legend_position = [10, 10]
        self.legend_size = [200, 50]
        self.dragging_legend = False
        self.legend_items = []

        self.face_symbols = True
        self.filled_symbols = True
        self.symbol_scale = 1
        self.transparency = 255
        self.grid = True

    def __del__(self):
        # TODO: delete shaders and vertex buffer
        glDeleteProgram(self.color_shader)

    def initializeGL(self):
        self.update_axes()
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)

        self.color_shader = glCreateProgram()
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        self.shaders = [vertex_shader, fragment_shader]

        vertex_shader_source = '''
            attribute vec3 position;
            attribute vec3 offset;
            attribute vec4 color;

            uniform bool face_symbols;
            uniform float symbol_scale;
            uniform float transparency;

            varying vec4 var_color;

            void main(void) {
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

              vec3 offset_rotated = offset;
              offset_rotated.x *= symbol_scale;
              offset_rotated.y *= symbol_scale;

              if (face_symbols)
                offset_rotated = invs * offset_rotated;

              vec4 off_pos = vec4(position+offset_rotated, 1);

              gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * off_pos;
              var_color = vec4(color.rgb, transparency);
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
        for shader, source in zip([vertex_shader, fragment_shader],
                                  [vertex_shader_source, fragment_shader_source]):
            glShaderSource(shader, 1, byref(source), byref(length))
            glCompileShader(shader)
            status = c_int()
            glGetShaderiv(shader, GL_COMPILE_STATUS, byref(status))
            if not status.value:
                print_log(shader)
                glDeleteShader(shader)
                return
            else:
                glAttachShader(self.color_shader, shader)

        glBindAttribLocation(self.color_shader, 0, 'position')
        glBindAttribLocation(self.color_shader, 1, 'offset')
        glBindAttribLocation(self.color_shader, 2, 'color')
        glLinkProgram(self.color_shader)
        self.color_shader_face_symbols = glGetUniformLocation(self.color_shader, 'face_symbols')
        self.color_shader_symbol_scale = glGetUniformLocation(self.color_shader, 'symbol_scale')
        self.color_shader_transparency = glGetUniformLocation(self.color_shader, 'transparency')
        linked = c_int()
        glGetProgramiv(self.color_shader, GL_LINK_STATUS, byref(linked))
        if not linked.value:
            print('Failed to link shader!')
        print('Shaders compiled and linked!')

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    def paintGL(self):
        glClearColor(1,1,1,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = self.width(), self.height()
        divide = self.zoom*10.
        if self.ortho:
            # TODO: fix ortho
            glOrtho(-width/divide, width/divide, -height/divide, height/divide, -1, 2000)
        else:
            aspect = float(width) / height if height != 0 else 1
            gluPerspective(30.0, aspect, 0.1, 2000)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        zoom = 100 if self.ortho else self.zoom
        gluLookAt(
            self.camera[0]*zoom + self.center[0],
            self.camera[1]*zoom + self.center[1],
            self.camera[2]*zoom + self.center[2],
            self.center[0],
            self.center[1],
            self.center[2],
            0, 1, 0)
        self.paint_axes()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for (cmd, params) in self.commands:
            if cmd == 'scatter':
                vao, vao_outline, array, labels = params
                glUseProgram(self.color_shader)
                glUniform1i(self.color_shader_face_symbols, self.face_symbols)
                glUniform1f(self.color_shader_symbol_scale, self.symbol_scale)
                glUniform1f(self.color_shader_transparency, self.transparency)
                if self.filled_symbols:
                    glBindVertexArray(vao.value)
                    glDrawArrays(GL_TRIANGLES, 0, vao.num_vertices)
                    glBindVertexArray(0)
                else:
                    glBindVertexArray(vao_outline.value)
                    glDrawElements(GL_LINES, vao_outline.num_indices, GL_UNSIGNED_INT, 0)
                    glBindVertexArray(0)
                glUseProgram(0)

                if labels != None:
                    for (x,y,z), label in zip(array, labels):
                        self.renderText(x,y,z, '{0:.2}'.format(label), font=self.labels_font)

        glDisable(GL_BLEND)
        if self.show_legend:
            self.draw_legend()

    def draw_legend(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        x, y = self.legend_position
        w, h = self.legend_size
        t = self.legend_border_thickness

        glDisable(GL_DEPTH_TEST)
        glColor4f(*self.legend_border_color)
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

        # TODO: clean this up
        glColor4f(0.1, 0.1, 0.1, 1)
        item_pos_y = y + 2*t + 10
        for shape, text in self.legend_items:
            if shape == 0:
                glBegin(GL_TRIANGLES)
                glVertex2f(x+10, item_pos_y)
                glVertex2f(x+20, item_pos_y)
                glVertex2f(x+15, item_pos_y-10)
                glEnd()

            self.renderText(x+20+4*t, item_pos_y, text)
            item_pos_y += 15

    def add_legend_item(self, shape, text):
        self.legend_items.append((shape, text))

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

    def paint_axes(self):
        zoom = 100 if self.ortho else self.zoom
        cam_in_space = numpy.array([
          self.center[0] + self.camera[0]*zoom,
          self.center[1] + self.camera[1]*zoom,
          self.center[2] + self.camera[2]*zoom
        ])

        def normal_from_points(p1, p2, p3):
            v1 = p2 - p1
            v2 = p3 - p1
            return normalize(numpy.cross(v1, v2))

        def plane_visible(plane):
            normal = normal_from_points(*plane[:3])
            cam_plane = normalize(plane[0] - cam_in_space)
            if numpy.dot(normal, cam_plane) > 0:
                return False
            return True

        def draw_line(line):
            glColor4f(0.2, 0.2, 0.2, 1)
            glLineWidth(2) # Widths > 1 are actually deprecated I think.
            glBegin(GL_LINES)
            glVertex3f(*line[0])
            glVertex3f(*line[1])
            glEnd()

        def draw_values(axis, coord_index, normal, sub=10):
            glColor4f(0.1, 0.1, 0.1, 1)
            glLineWidth(1)
            start = axis[0]
            end = axis[1]
            direction = (end - start) / float(sub)
            middle = (end - start) / 2.
            offset = normal*0.3
            for i in range(sub):
                position = start + direction*i
                glBegin(GL_LINES)
                glVertex3f(*(position-normal*0.08))
                glVertex3f(*(position+normal*0.08))
                glEnd()
                position += offset
                self.renderText(position[0],
                                position[1],
                                position[2],
                                '{0:.2}'.format(position[coord_index]))

        glDisable(GL_DEPTH_TEST)
        glLineWidth(1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        bb_center = (self.b_box[1] + self.b_box[0]) / 2.

        # Draw axis labels.
        glColor4f(0,0,0,1)
        for axis, title in zip([self.x_axis, self.y_axis, self.z_axis],
                               [self.x_axis_title, self.y_axis_title, self.z_axis_title]):
            middle = (axis[0] + axis[1]) / 2.
            self.renderText(middle[0], middle[1]-0.2, middle[2]-0.2, title,
                            font=self.axis_title_font)

        def draw_axis_plane(axis_plane, sub=10):
            normal = normal_from_points(*axis_plane[:3])
            camera_vector = normalize(axis_plane[0] - cam_in_space)
            cos = numpy.dot(normal, camera_vector)
            cos = max(0.7, cos)
            glColor4f(*(self.color_plane*cos))
            P11, P12, P21, P22 = numpy.asarray(axis_plane)
            glBegin(GL_QUADS)
            glVertex3f(*P11)
            glVertex3f(*P12)
            glVertex3f(*P21)
            glVertex3f(*P22)
            glEnd()
            P22, P21 = P21, P22
            glColor4f(*(self.color_grid * cos))
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

        planes = [self.axis_plane_xy, self.axis_plane_yz,
                  self.axis_plane_xy_back, self.axis_plane_yz_right]
        visible_planes = map(plane_visible, planes)
        if self.grid:
            draw_axis_plane(self.axis_plane_xz)
            for visible, plane in zip(visible_planes, planes):
                if not visible:
                    draw_axis_plane(plane)

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

        if visible_planes[0]:
            draw_line(self.x_axis)
            draw_values(self.x_axis, 0, numpy.array([0,0,-1]))
        elif visible_planes[2]:
            draw_line(self.x_axis + self.unit_z)
            draw_values(self.x_axis + self.unit_z, 0, numpy.array([0,0,1]))

        if visible_planes[1]:
            draw_line(self.z_axis)
            draw_values(self.z_axis, 2, numpy.array([-1,0,0]))
        elif visible_planes[3]:
            draw_line(self.z_axis + self.unit_x)
            draw_values(self.z_axis + self.unit_x, 2, numpy.array([1,0,0]))

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
        normals = [numpy.array([1,0,0]),
                   numpy.array([0,0,1]),
                   numpy.array([-1,0,0]),
                   numpy.array([0,0,-1])
                ]
        axis = y_axis_translated[rightmost_visible]
        draw_line(axis)
        normal = normals[rightmost_visible]
        draw_values(y_axis_translated[rightmost_visible], 1, normal)

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

    def scatter(self, X, Y, Z, colors='b', sizes=5, shapes=None, labels=None, **kwargs):
        array = [[x, y, z] for x,y,z in zip(X, Y, Z)]
        if isinstance(colors, str):
            color_map = {'r': [1.0, 0.0, 0.0, 1.0],
                         'g': [0.0, 1.0, 0.0, 1.0],
                         'b': [0.0, 0.0, 1.0, 1.0]}
            default = [0.0, 0.0, 1.0, 1.0]
            colors = [color_map.get(colors, default) for _ in array]
 
        if isinstance(sizes, int):
            sizes = [sizes for _ in array]

        if shapes == None:
            shapes = [0 for _ in array]

        max, min = numpy.max(array, axis=0), numpy.min(array, axis=0)
        self.b_box = [max, min]
        self.minx, self.miny, self.minz = min
        self.maxx, self.maxy, self.maxz = max
        self.center = (min + max) / 2 
        self.normal_size = numpy.max(self.center - self.b_box[1]) / 100.

        # TODO: more shapes? For now, wrap other to these ones.
        different_shapes = [3, 4, 5, 8]

        # Generate vertices for shapes and also indices for outlines.
        vertices = []
        outline_indices = []
        index = 0
        for (x,y,z), (r,g,b,a), size, shape in zip(array, colors, sizes, shapes):
            sO2 = size * self.normal_size / 2.
            n = different_shapes[shape % len(different_shapes)]
            angle_inc = 2.*pi / n
            angle = angle_inc / 2.
            for i in range(n):
                vertices.extend([x,y,z, 0,0,0, r,g,b,a])
                vertices.extend([x,y,z, -cos(angle)*sO2, -sin(angle)*sO2, 0, r,g,b,a])
                angle += angle_inc
                vertices.extend([x,y,z, -cos(angle)*sO2, -sin(angle)*sO2, 0, r,g,b,a])
                outline_indices.extend([index+1, index+2])
                index += 3

        # Build Vertex Buffer + Vertex Array Object.
        vao = c_int()
        glGenVertexArrays(1, byref(vao))
        glBindVertexArray(vao.value)

        vertex_buffer = c_int()
        glGenBuffers(1, byref(vertex_buffer))
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer.value)

        vertex_size = (3+3+4)*4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, 3*4)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, vertex_size, 6*4)
        glEnableVertexAttribArray(2)

        # It's important to keep a reference to vertices around,
        # data uploaded to GPU seem to get corrupted otherwise.
        vertex_buffer.vertices = numpy.array(vertices, 'f')
        glBufferData(GL_ARRAY_BUFFER,
            ArrayDatatype.arrayByteCount(vertex_buffer.vertices),
            ArrayDatatype.voidDataPointer(vertex_buffer.vertices), GL_STATIC_DRAW)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Outline:
        # generate another VAO, keep the same vertex buffer, but use an index buffer
        # this time.
        index_buffer = c_int()
        glGenBuffers(1, byref(index_buffer))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer.value)
        index_buffer.indices = numpy.array(outline_indices, 'I')
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
            ArrayDatatype.arrayByteCount(index_buffer.indices),
            ArrayDatatype.voidDataPointer(index_buffer.indices), GL_STATIC_DRAW)

        vao_outline = c_int()
        glGenVertexArrays(1, byref(vao_outline))
        glBindVertexArray(vao_outline.value)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer.value)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer.value)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, 3*4)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, vertex_size, 6*4)
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        vao.num_vertices = len(vertices) / (vertex_size / 4)
        vao_outline.num_indices = vao.num_vertices * 2 / 3
        self.vertex_buffers.append(vertex_buffer)
        self.index_buffers.append(index_buffer)
        self.vaos.append(vao)
        self.vaos.append(vao_outline)
        self.commands.append(("scatter", [vao, vao_outline, array, labels]))
        self.update_axes()
        self.updateGL()

    def mousePressEvent(self, event):
      self.mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        dx = pos.x() - self.mouse_pos.x()
        dy = pos.y() - self.mouse_pos.y()

        if event.buttons() & Qt.LeftButton:
            if self.dragging_legend:
                self.legend_position[0] += dx
                self.legend_position[1] += dy
            elif self.legend_position[0] <= pos.x() <= self.legend_position[0]+self.legend_size[0] and\
               self.legend_position[1] <= pos.y() <= self.legend_position[1]+self.legend_size[1]:
                self.dragging_legend = True
        elif event.buttons() & Qt.MiddleButton:
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

    def mouseReleaseEvent(self, event):
        self.dragging_legend = False

    def wheelEvent(self, event):
        if event.orientation() == Qt.Vertical:
            self.zoom -= event.delta() / self.zoom_factor
            if self.zoom < 2:
                self.zoom = 2
            self.updateGL()

    def clear(self):
        self.commands = []
        self.legend_items = []


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

    w.scatter(x, y, z, colors=colors)
    app.exec_()
