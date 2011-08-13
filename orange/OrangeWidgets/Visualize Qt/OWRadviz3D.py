"""
<name>Sphereviz 3D</name>
<icon>icons/Radviz.png</icon>
<priority>220</priority>
"""

import os

from plot.owplot3d import *
from plot.owprimitives3d import parse_obj
from plot.owplotgui import OWPlotGUI
from OWLinProjQt import *

class OWRadviz3DPlot(OWPlot3D, orngScaleLinProjData3D):
    def __init__(self, widget, parent=None, name='Sphereviz 3D'):
        OWPlot3D.__init__(self, parent)
        orngScaleLinProjData3D.__init__(self)

        self.camera_fov = 40.
        self.show_axes = self.show_chassis = self.show_grid = False

        self.point_width = 5
        self.alpha_value = 255
        self.show_filled_symbols = True
        self.use_antialiasing = True
        self.sendSelectionOnUpdate = False
        self.setCanvasBackground = self.setCanvasColor

        self.gui = OWPlotGUI(self)

    def setData(self, data, subsetData=None, **args):
        orngScaleLinProjData3D.setData(self, data, subsetData, **args)

    def updateData(self, labels=None, setAnchors=0, **args):
        self.clear()

        if not self.haveData or len(labels) < 3:
            self.anchor_data = []
            self.updateLayout()
            return

        # TODO: do this once (in constructor?)
        sphere_data = parse_obj(os.path.join(os.path.dirname(__file__), '../plot/primitives/sphere_hq.obj'))
        vertices = []
        for v0, v1, v2, n0, n1, n2 in sphere_data:
            vertices.extend([v0[0],v0[1],v0[2], n0[0],n0[1],n0[2],
                             v1[0],v1[1],v1[2], n1[0],n1[1],n1[2],
                             v2[0],v2[1],v2[2], n2[0],n2[1],n2[2]])

        self.sphere_vao_id = GLuint(0)
        glGenVertexArrays(1, self.sphere_vao_id)
        glBindVertexArray(self.sphere_vao_id)

        vertex_buffer_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, numpy.array(vertices, 'f'), GL_STATIC_DRAW)

        vertex_size = (3+3)*4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(3*4))
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.sphere_vao_id.num_vertices = len(vertices) / (vertex_size / 4)

        self.sphere_shader = QtOpenGL.QGLShaderProgram()
        vertex_shader_source = '''
            #extension GL_EXT_gpu_shader4 : enable

            attribute vec3 position;
            attribute vec3 normal;

            uniform vec3 color;

            varying vec4 var_color;

            void main(void) {
                gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(position*8.6602, 1.);
                var_color = vec4(0.8, 0.8, 0.8, 0.2);
            }
            '''

        fragment_shader_source = '''
            varying vec4 var_color;

            void main(void) {
              gl_FragColor = var_color;
            }
            '''

        self.sphere_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self.sphere_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self.sphere_shader.bindAttributeLocation('position', 0)
        self.sphere_shader.bindAttributeLocation('normal',   1)

        if not self.sphere_shader.link():
            print('Failed to link sphere shader!')
        else:
            print('Sphere shader linked.')

        if setAnchors:
            self.setAnchors(args.get('XAnchors'), args.get('YAnchors'), args.get('ZAnchors'), labels)

        data_size = len(self.rawData)
        indices = [self.attributeNameIndex[anchor[3]] for anchor in self.anchor_data]
        valid_data = self.getValidList(indices)
        trans_proj_data = self.createProjectionAsNumericArray(indices, validData=valid_data,
            scaleFactor=self.scaleFactor, normalize=self.normalizeExamples, jitterSize=-1,
            useAnchorData=1, removeMissingData=0)
        if trans_proj_data == None:
            return
        proj_data = trans_proj_data.T
        x_positions = proj_data[0]
        y_positions = proj_data[1]
        z_positions = proj_data[2]

        self.scatter(x_positions, y_positions, z_positions)

        self.commands.append(('custom', self.draw_sphere_callback))
        self.updateGL()

    def updateGraph(self, attrList=None, setAnchors=0, insideColors=None, **args):
        pass

    def draw_sphere_callback(self):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.sphere_shader.bind()
        glBindVertexArray(self.sphere_vao_id)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glDrawArrays(GL_TRIANGLES, 0, self.sphere_vao_id.num_vertices)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glBindVertexArray(0)
        self.sphere_shader.release()

        radius = (25*3)**0.5

        if self.showAnchors:
            for anchor in self.anchor_data:
                x, y, z, label = anchor
                glLineWidth(2.)
                glColor4f(0.8, 0, 0, 1)
                glBegin(GL_LINES)
                glVertex3f(x*0.98*radius, y*0.99*radius, z*0.98*radius)
                glVertex3f(x*1.02*radius, y*1.02*radius, z*1.02*radius)
                glEnd()
                glLineWidth(1.)

                glColor4f(0, 0, 0, 1)
                self.renderText(x*1.1*radius, y*1.1*radius, z*1.1*radius, label)

    def setCanvasColor(self, c):
        pass

    def getSelectionsAsExampleTables(self, attrList, useAnchorData=1, addProjectedPositions=0):
        return (None, None)

    def removeAllSelections(self):
        pass

    def replot(self):
        pass

    # TODO: catch mouseEvents

class OWRadviz3D(OWLinProjQt):
    settingsList = ['showAllAttributes']

    def __init__(self, parent=None, signalManager=None):
        OWLinProjQt.__init__(self, parent, signalManager, "Radviz", graphClass=OWRadviz3DPlot)

        self.inputs = [("Examples", ExampleTable, self.setData, Default),
                       ("Example Subset", ExampleTable, self.setSubsetData),
                       ("Attribute Selection List", AttributeList, self.setShownAttributes),
                       ("Evaluation Results", orngTest.ExperimentResults, self.setTestResults),
                       ("VizRank Learner", orange.Learner, self.setVizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTable),
                        ("Unselected Examples", ExampleTable),
                        ("Attribute Selection List", AttributeList)]
        self.resize(1000, 600)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    radviz = OWRadviz3D()
    radviz.show()
    data = orange.ExampleTable('../../doc/datasets/iris')
    radviz.setData(data)
    radviz.handleNewSignals()
    app.exec_()
