from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtOpenGL

import orange

from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import numpy
from math import sin, cos

def normalize(vec):
  return vec / numpy.sqrt(numpy.sum(vec** 2))

class OWGraph3D(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers), parent)
        self.commands = []
        self.minx = self.miny = self.minz = 0
        self.maxx = self.maxy = self.maxz = 0
        self.b_box = [numpy.array([0, 0, 0]), numpy.array([0, 0, 0])]
        self.camera = numpy.array([0.6, 0.8, 0])  # Spherical unit vector around the center. This is where camera is looking from.
        self.center = numpy.array([0, 0, 0])      # Camera is looking into this point.

        # Try to use displays lists for performance.
        self.sphere_dl = glGenLists(1) # TODO: why does this fail?
        if self.sphere_dl != 0:
          gluQuadric  = gluNewQuadric()
          glNewList(self.sphere_dl, GL_COMPILE)
          gluSphere(gluQuadric, 1, 10, 10)
          glEndList()
          gluDeleteQuadric(gluQuadric)

        # TODO: other shapes

        self.yaw = self.pitch = 0
        self.rotation_factor = 100.
        self.zoom_factor = 100.
        self.zoom = 10
        self.move_factor = 100.
        self.mouse_pos = [100,100] # TODO: get real mouse position, calculate camera, fix the initial jump
        self.updateAxes()

        self.axisTitleFont = QFont('Helvetica', 10, QFont.Bold)
        self.ticksFont = QFont('Helvetica', 9)
        self.XaxisTitle = ''
        self.YaxisTitle = ''
        self.ZaxisTitle = ''

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
 
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if h == 0:
            aspect = 1
        else:
            aspect = float(w)/float(h)
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
        self.paintAxes()

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
 
        if self.sphere_dl == 0:
          gluQuadric  = gluNewQuadric()
        for cmd, (array, colors, sizes) in self.commands:
            for (x,y,z), (r, g, b, a), size in zip(array, colors, sizes):
                glPushMatrix()
                glTranslatef(x, y, z)
                glColor4f(r, g, b, a)
                scale = self.normalSize * size
                glScalef(scale, scale, scale)
                if self.sphere_dl == 0:
                  gluSphere(gluQuadric, 1, 10, 10)
                else:
                  glCallList(self.sphere_dl)
                glPopMatrix()
        if self.sphere_dl == 0:
          gluDeleteQuadric(gluQuadric)

        glDisable(GL_CULL_FACE)

    def setXaxisTitle(self, title):
      self.XaxisTitle = title
      self.updateGL()

    def setYaxisTitle(self, title):
      self.YaxisTitle = title
      self.updateGL()

    def setZaxisTitle(self, title):
      self.ZaxisTitle = title
      self.updateGL()

    def paintAxes(self):
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
        self.renderText(ac[0], ac[1]-0.2, ac[2]-0.2, self.XaxisTitle)
        ac = (self.y_axis[0] + self.y_axis[1]) / 2.
        self.renderText(ac[0], ac[1]-0.2, ac[2]-0.2, self.YaxisTitle)
        ac = (self.z_axis[0] + self.z_axis[1]) / 2.
        self.renderText(ac[0], ac[1]-0.2, ac[2]-0.2, self.ZaxisTitle)

        outwards = normalize(self.x_axis[0] - bb_center)
        pos = self.x_axis[0] + outwards * 0.2
        self.renderText(pos[0], pos[1], pos[2], '{0:.2}'.format(pos[0]))

        glColor4f(1,1,1,1)

        def paintGrid(planeQuad, sub=20):
            P11, P12, P22, P21 = numpy.asarray(planeQuad)
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

        def paintQuad(planeQuad):
            P11, P12, P21, P22 = numpy.asarray(planeQuad)
            glBegin(GL_QUADS)
            glVertex3f(*P11)
            glVertex3f(*P12)
            glVertex3f(*P21)
            glVertex3f(*P22)
            glEnd()

        colorPlane = [0.5, 0.5, 0.5, 0.5]
        colorGrid = [0.3, 0.3, 0.3, 1.0]

        def paintPlain(planeQuad):
            #glColor4f(*colorPlane)
            #paintQuad(planeQuad)
            glColor4f(*colorGrid)
            paintGrid(planeQuad)

        def normalFromPoints(P1, P2, P3):
            V1 = P2 - P1
            V2 = P3 - P1
            return normalize(numpy.cross(V1, V2))
 
        def drawGridVisible(planeQuad, ccw=False):
            normal = normalFromPoints(*planeQuad[:3])
            camInSpace = numpy.array([
              self.center[0] + self.camera[0]*self.zoom,
              self.center[1] + self.camera[1]*self.zoom,
              self.center[2] + self.camera[2]*self.zoom
            ])
            cameraVector = normalize(planeQuad[0] - camInSpace)
            cos = numpy.dot(normal, cameraVector) * (-1 if ccw else 1)
            if cos > 0:
                paintPlain(planeQuad)

        drawGridVisible(self.axisPlaneXY)
        drawGridVisible(self.axisPlaneYZ)
        drawGridVisible(self.axisPlaneXZ)
        drawGridVisible(self.axisPlaneXYBack)
        drawGridVisible(self.axisPlaneYZRight)
        drawGridVisible(self.axisPLaneXZTop)

        glEnable(GL_CULL_FACE)

    def updateAxes(self):
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

        self.axisPlaneXY = [A, B, C, D]
        self.axisPlaneYZ = [A, D, H, E]
        self.axisPlaneXZ = [D, C, G, H]

        self.axisPlaneXYBack = [H, G, F, E]
        self.axisPlaneYZRight = [B, F, G, C]
        self.axisPLaneXZTop = [E, F, B, A]

    def scatter(self, X, Y, Z=0, c="b", s=20, **kwargs):
        array = [[x, y, z] for x,y,z in zip(X, Y, Z)]
        if isinstance(c, str):
            colorDict ={"r": [1.0, 0.0, 0.0, 1.0],
                        "g": [0.0, 1.0, 0.0, 1.0],
                        "b": [0.0, 0.0, 1.0, 1.0]}
            default = [0.0, 0.0, 1.0, 1.0]
            colors = [colorDict.get(c, default) for i in array]
        else:
            colors = c
 
        if isinstance(s, int):
            s = [s for _ in array]

        self.commands.append(("scatter", (array, colors, s)))
        max, min = numpy.max(array, axis=0), numpy.min(array, axis=0)
        self.b_box = [max, min]
        self.minx, self.miny, self.minz = min
        self.maxx, self.maxy, self.maxz = max
        self.center = (min + max) / 2 
        self.normalSize = numpy.max(self.center - self.b_box[1]) / 100.
        self.updateAxes()
        self.updateGL()
 
    def mousePressEvent(self, event):
      self.mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
      if event.buttons() & Qt.MiddleButton:
        pos = event.pos()
        dx = pos.x() - self.mouse_pos.x()
        dy = pos.y() - self.mouse_pos.y()
        if QApplication.keyboardModifiers() & Qt.ShiftModifier:
          off = numpy.cross(self.center - self.camera, [0,1,0]) * (dx / self.move_factor)
          self.center -= off
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
        self.zoom += event.delta() / self.zoom_factor
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
    rand = lambda :random() - 0.5
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
