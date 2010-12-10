
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtOpenGL

import orange

import OpenGL
OpenGL.FORWARD_COMPATIBLE_ONLY = True
 
import OpenGL.GL as gl
import OpenGL.GLU as glu
import sys, os
import numpy

class BufferObject(object):
    def __init__(self, glId=None, data=None):
        self.glId = glId
        self.data = data
        if glId is None:
            self.glId = self.genBuffer()
            
    @classmethod
    def genBuffer(cls):
        id = gl.glGenBuffers(1)
        return id[0]
        

class OWGraph3D(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, QtOpenGL.QGLFormat(QtOpenGL.QGL.SampleBuffers), parent)
        self.commands = []
        
        self.quad = numpy.array([[-1., 1., 0],
                     [1., 1., 0],
                     [1., -1., 0],
                     [-1., -1, 0]], dtype="f")
        
        self.minx = self.miny = self.minz = 0
        self.maxx = self.maxy = self.maxz = 0
        self.b_box = [0, 0, 0], [0, 0, 0]
        self.camera = numpy.array([0, 0, 0])
        self.center = numpy.array([2, 0, 0])
        
    def initializeGL(self):
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClearDepth(1.0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glShadeModel(gl.GL_SMOOTH)
#        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45.0, float(self.width())/float(self.height()), 0.1, 100.0)
        glu.gluOrtho2D(0, 10, 0, 10)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
    def resizeGL(self, w, h):
        print "Resize"
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        if h == 0:
            aspect = 1
        else:
            aspect = float(w)/float(h)
        glu.gluPerspective(45.0, aspect, 0.1, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
#        gl.glPushMatrix()
        glu.gluLookAt(self.camera[0], self.camera[1], self.camera[2],
                      self.center[0], self.center[1], self.center[2],
                      0, 1, 0)
        self.paintAxes()
        
        size = numpy.max(self.center - self.b_box[1]) * 2 / 50.0  
        print size
        
        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gluQuadric  = glu.gluNewQuadric()
        for cmd, (array, colors) in self.commands:
            for (x,y,z), (r, g, b) in zip(array, colors):
                gl.glPushMatrix()
                gl.glTranslatef(x, y, z)
                gl.glColor4f(r, g, b, 0.3)
                glu.gluSphere(gluQuadric, size, 5, 5)
                gl.glPopMatrix()
        glu.gluDeleteQuadric(gluQuadric)
        
                
    def paintGLScatter(self, array, colors="r", shape=".", size=1):
        for x, y, z in array:
            pass
            
        
    def paintAxes(self):
        x_axis = [[self.minx, self.miny, self.minz],
                  [self.maxx, self.miny, self.minz]]
        y_axis = [[self.minx, self.miny, self.minz],
                  [self.minx, self.maxy, self.minz]]
        z_axis = [[self.minx, self.miny, self.minz],
                  [self.minx, self.miny, self.maxz]]
        x_axis = numpy.array(x_axis)
        y_axis = numpy.array(y_axis)
        z_axis = numpy.array(z_axis)
        
        unit_x = numpy.array([self.maxx - self.minx, 0, 0])
        unit_y = numpy.array([0, self.maxy - self.miny, 0])
        unit_z = numpy.array([0, 0, self.maxz - self.minz])
        
        A = y_axis[1]
        B = y_axis[1] + unit_x
        C = x_axis[1]
        D = x_axis[0]
        
        E = A + unit_z
        F = B + unit_z
        G = C + unit_z
        H = D + unit_z
        
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glColor4f(1,1,1,1)
        for start, end in [x_axis, y_axis, z_axis]:
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(*start)
            gl.glVertex3f(*end)
            gl.glEnd()
            
        def paintGrid(planeQuad, sub=4):
            P11, P12, P22, P21 = numpy.asarray(planeQuad)
            Dx = numpy.linspace(0.0, 1.0, num=sub)
            P1vecH = P12 - P11
            P2vecH = P22 - P21
            P1vecV = P21 - P11
            P2vecV = P22 - P12
            gl.glBegin(gl.GL_LINES)
            for i, dx in enumerate(Dx):
                start = P11 + P1vecH*dx
                end = P21 + P2vecH*dx
                gl.glVertex3f(*start)
                gl.glVertex3f(*end)
                
                start = P11 + P1vecV*dx
                end = P12 + P2vecV*dx
                gl.glVertex3f(*start)
                gl.glVertex3f(*end)
            gl.glEnd()
            
        def paintQuad(planeQuad):
            P11, P12, P21, P22 = numpy.asarray(planeQuad)
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(*P11)
            gl.glVertex3f(*P12)
            gl.glVertex3f(*P21)
            gl.glVertex3f(*P22)
            gl.glEnd()
            
        colorPlane = [0.5, 0.5, 0.5, 0.5]
        colorGrid = [0.0, 0.0, 0.0, 1.0]
        
        def paintPlain(planeQuad):
            gl.glColor4f(*colorPlane)
            paintQuad(plane)
            gl.glColor4f(*colorGrid)
            paintGrid(plane)
            
        
        def normalize(Vec):            
            return Vec / numpy.sqrt(numpy.sum(Vec ** 2))
            
        def normalFromPoints(P1, P2, P3):
            V1 = P2 - P1
            V2 = P3 - P1
            return normalize(numpy.cross(V1, V2))
        
        cameraVector = normalize(self.center - self.camera)
#        cameraVector *= numpy.sqrt(numpy.sum(cameraVector ** 2)
        def paintPlainIf(planeQuad, ccw=False):
            normal = normalFromPoints(*planeQuad[:3])
            cameraVector = planeQuad[0] - self.camera
            cos = numpy.dot(normal, cameraVector) * (-1 if ccw else 1)
            if cos > 0:
                paintPlain(planeQuad)
                
        #xy plane
        plane = [A, B, C, D]
        paintPlainIf(plane)

        # yz plane
        plane = [A, D, H, E]
        paintPlainIf(plane)
        
        # xz plane
        plane = [D, C, G, H]
        paintPlainIf(plane)
        
        # xy back plane
        plane = [H, G, F, E]
        paintPlainIf(plane)
        
        # yz right plane
        plane = [B, F, G, C]
        paintPlainIf(plane)
        
        # xz top plane
        plane = [E, F, B, A]
        paintPlainIf(plane)
        
        gl.glEnable(gl.GL_CULL_FACE)
        
        
    def plot3d(self, x, y, z, format="."):
        pass
    
    def scatter3d(self, X, Y, Z, c="b", marker="o"):
        array = [[x, y, z] for x,y,z in zip(X, Y, Z)]
        if isinstance(c, str):
            colorDict ={"r": [1.0, 0.0, 0.0],
                        "g": [0.0, 1.0, 0.0],
                        "b": [0.0, 0.0, 1.0]}
            default = [0.0, 0.0, 1.0]
            colors = [colorDict.get(c, default) for i in array]
        else:
            colors = c
            
        self.commands.append(("scatter", (array, colors)))
        max, min = numpy.max(array, axis=0), numpy.min(array, axis=0)
        self.b_box = max, min
        self.minx, self.miny, self.minz = min
        self.maxx, self.maxy, self.maxz = max
        self.center = (min + max) / 2 
        self.updateGL()
        
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_camera = numpy.array(self.camera)
            self._lastMousePos = glu.gluUnProject(event.x(), event.y(), 0)
            
        
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            x, y, z = glu.gluUnProject(event.x(), event.y(), 0)
            
            radius = self.radius()
            
            def normalize(P):
                return P / numpy.sqrt(numpy.sum(P ** 2))
            
            def toSphere(P, r=radius):
                #Project P to the centered sphere
                P = P - self.center
                P = normalize(P)
                return self.center + (P * radius)
                
            diff = numpy.array([x, y, z]) - numpy.array(self._lastMousePos)
            pos =  diff + self._last_camera
            
            camera = numpy.array(self._last_camera)
            center = numpy.array(self.center)
            dist = numpy.sqrt(numpy.sum((camera - center) ** 2)) #distance from the center
            camera = (pos - center) / numpy.sqrt(numpy.sum((pos - center) ** 2))
            camera = camera * dist
            camera = center + camera
            self.camera[:] = camera
            
            self._lastMousePos = x, y, z
            self._last_camera = numpy.array(self.camera) 
            self.updateGL()
            
            
    def radius(self):
        return numpy.max(self.center - self.b_box[1]) * 2  
        
            
    def wheelEvent(self, event):
        if event.orientation() == Qt.Vertical:
            degrees = event.delta() / 8
            steps = degrees / 15
            cameraToCenter = self.camera - self.center
            radius = numpy.sqrt(numpy.sum((self.center - self.b_box[0]) ** 2))
            cameraToCenter = cameraToCenter / numpy.sqrt(numpy.sum(cameraToCenter ** 2))
            diff = cameraToCenter * (radius / 360) * degrees * 8
            self.camera = self.camera + diff
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
    colors = [[c.red()/255., c.green()/255., c.blue()/255.] for c in colors]

#    x = [rand()*2 for i in range(N)]
#    y = [rand()*2 for i in range(N)]
#    z = [-3 + rand() for i in range(N)]
#    colors = "b"
    w.scatter3d(x, y, z, c=colors)
    app.exec_()
        