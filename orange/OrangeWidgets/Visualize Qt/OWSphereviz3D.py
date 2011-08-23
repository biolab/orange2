'''
<name>Sphereviz 3D</name>
<icon>icons/Radviz.png</icon>
<priority>2000</priority>
'''

import os

from plot.owplot3d import *
from plot.primitives import parse_obj
from plot.owplotgui import OWPlotGUI
from OWLinProjQt import *

from Orange.preprocess.scaling import ScaleLinProjData3D

import orange
Discrete = orange.VarTypes.Discrete
Continuous = orange.VarTypes.Continuous

class OWSphereviz3DPlot(OWPlot3D, ScaleLinProjData3D):
    def __init__(self, widget, parent=None, name='None'):
        OWPlot3D.__init__(self, parent)
        ScaleLinProjData3D.__init__(self)

        self.camera_fov = 50.
        self.show_axes = self.show_chassis = self.show_grid = False

        self.point_width = 6
        self.animate_plot = False
        self.animate_points = False
        self.antialias_plot = False
        self.antialias_points = False
        self.antialias_lines = False
        self.auto_adjust_performance = False
        self.show_filled_symbols = True
        self.use_antialiasing = True
        self.sendSelectionOnUpdate = False
        self.setCanvasBackground = self.setCanvasColor

        self._point_width_to_symbol_scale = 1.5
        self.camera_in_center = False

        self.gui = OWPlotGUI(self)

    def setData(self, data, subsetData=None, **args):
        ScaleLinProjData3D.setData(self, data, subsetData, **args)
        self.initializeGL() # Apparently this is not called already
        self.makeCurrent()

        if hasattr(self, 'sphere_vao_id'):
            return

        sphere_data = parse_obj('sphere_hq.obj')
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

        cone_data = parse_obj('cone_hq.obj')
        vertices = []
        for v0, v1, v2, n0, n1, n2 in cone_data:
            vertices.extend([v0[0],v0[1],v0[2], n0[0],n0[1],n0[2],
                             v1[0],v1[1],v1[2], n1[0],n1[1],n1[2],
                             v2[0],v2[1],v2[2], n2[0],n2[1],n2[2]])

        self.cone_vao_id = GLuint(0)
        glGenVertexArrays(1, self.cone_vao_id)
        glBindVertexArray(self.cone_vao_id)

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

        self.cone_vao_id.num_vertices = len(vertices) / (vertex_size / 4)

        # Geometry shader-based wireframe rendering
        # (http://cgg-journal.com/2008-2/06/index.html)
        self.sphere_shader = QtOpenGL.QGLShaderProgram()
        geometry_shader_source = '''#version 150
            layout(triangles) in;
            layout(triangle_strip, max_vertices=3) out;

            uniform mat4 projection;
            uniform mat4 modelview;
            uniform vec3 cam_position;
            out float transparency;

            uniform vec2 win_scale;
            noperspective out vec3 dist;

            void main(void)
            {
                vec4 pos_in0 = projection * modelview * gl_PositionIn[0];
                vec4 pos_in1 = projection * modelview * gl_PositionIn[1];
                vec4 pos_in2 = projection * modelview * gl_PositionIn[2];
                vec2 p0 = win_scale * pos_in0.xy / pos_in0.w;
                vec2 p1 = win_scale * pos_in1.xy / pos_in1.w;
                vec2 p2 = win_scale * pos_in2.xy / pos_in2.w;

                vec2 v0 = p2-p1;
                vec2 v1 = p2-p0;
                vec2 v2 = p1-p0;
                float area = abs(v1.x*v2.y - v1.y * v2.x);

                dist = vec3(area / length(v0), 0., 0.);
                gl_Position = pos_in0;
                transparency = clamp(dot(normalize(cam_position-gl_PositionIn[0].xyz), normalize(gl_PositionIn[0].xyz)), 0., 1.);
                EmitVertex();

                dist = vec3(0., area / length(v1), 0.);
                gl_Position = pos_in1;
                transparency = clamp(dot(normalize(cam_position-gl_PositionIn[1].xyz), normalize(gl_PositionIn[1].xyz)), 0., 1.);
                EmitVertex();

                dist = vec3(0., 0., area / length(v2));
                gl_Position = pos_in2;
                transparency = clamp(dot(normalize(cam_position-gl_PositionIn[2].xyz), normalize(gl_PositionIn[2].xyz)), 0., 1.);
                EmitVertex();

                EndPrimitive();
            }
        '''

        vertex_shader_source = '''#version 150
            attribute vec3 position;
            attribute vec3 normal;

            void main(void)
            {
                gl_Position = vec4(position, 1.);
            }
            '''

        fragment_shader_source = '''#version 150
            in float transparency;
            noperspective in vec3 dist;

            uniform bool invert_transparency;
            const vec4 wire_color = vec4(0.5, 0.5, 0.5, 0.5);
            const vec4 fill_color = vec4(1., 1., 1., 0.5);

            void main(void)
            {
                float d = min(dist[0], min(dist[1], dist[2]));
                gl_FragColor = mix(wire_color, fill_color, 1. - exp2(-1*d*d));
                if (invert_transparency)
                    gl_FragColor.a = transparency;
                else
                    gl_FragColor.a = 1. - transparency - 0.6;
            }
            '''

        self.sphere_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Geometry, geometry_shader_source)
        self.sphere_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self.sphere_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self.sphere_shader.bindAttributeLocation('position', 0)
        self.sphere_shader.bindAttributeLocation('normal',   1)

        if not self.sphere_shader.link():
            print('Failed to link sphere shader!')

        vertex_shader_source = '''#version 150
            in vec3 position;
            in vec3 normal;

            out vec4 color;

            uniform mat4 projection;
            uniform mat4 modelview;

            const vec3 light_direction = normalize(vec3(-0.7, 0.42, 0.21));

            void main(void)
            {
                gl_Position = projection * modelview * vec4(position, 1.);
                float diffuse = clamp(dot(light_direction, normalize((modelview * vec4(normal, 0.)).xyz)), 0., 1.);
                color = vec4(vec3(0., 1., 0.) * diffuse, 1.);
            }
            '''
        fragment_shader_source = '''#version 150
            in vec4 color;

            void main(void)
            {
                gl_FragColor = color;
            }
            '''

        self.cone_shader = QtOpenGL.QGLShaderProgram()
        self.cone_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self.cone_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self.cone_shader.bindAttributeLocation('position', 0)
        self.cone_shader.bindAttributeLocation('normal', 1)

        if not self.cone_shader.link():
            print('Failed to link cone shader!')

    def updateData(self, labels=None, setAnchors=0, **args):
        self.clear()

        if not self.have_data or len(labels) < 3:
            self.anchor_data = []
            self.updateGL()
            return

        if setAnchors:
            self.setAnchors(args.get('XAnchors'), args.get('YAnchors'), args.get('ZAnchors'), labels)

        indices = [self.attribute_name_index[anchor[3]] for anchor in self.anchor_data]
        valid_data = self.getValidList(indices)
        trans_proj_data = self.create_projection_as_numeric_array(indices, validData=valid_data,
            scaleFactor=1.0, normalize=self.normalizeExamples, jitterSize=-1,
            useAnchorData=1, removeMissingData=0)
        if trans_proj_data == None:
            return

        proj_data = trans_proj_data.T
        proj_data[0:3] += 0.5 # Geometry shader offsets positions by -0.5; leave class unmodified
        if self.data_has_discrete_class:
            proj_data[3] = self.no_jittering_scaled_data[self.attribute_name_index[self.data_domain.classVar.name]]
        self.set_plot_data(proj_data, None)
        self.symbol_scale = self.point_width*self._point_width_to_symbol_scale
        self.hide_outside = False
        self.fade_outside = False

        color_index = symbol_index = size_index = label_index = -1
        color_discrete = False
        x_discrete = self.data_domain[self.anchor_data[0][3]].varType == Discrete
        y_discrete = self.data_domain[self.anchor_data[1][3]].varType == Discrete
        z_discrete = self.data_domain[self.anchor_data[2][3]].varType == Discrete

        if self.data_has_discrete_class:
            self.discPalette.setNumberOfColors(len(self.dataDomain.classVar.values))

        use_different_symbols = self.useDifferentSymbols and self.data_has_discrete_class and\
            len(self.data_domain.classVar.values) < len(Symbol)

        if use_different_symbols:
            symbol_index = 3
            num_symbols_used = len(self.data_domain.classVar.values)
        else:
            num_symbols_used = -1

        if self.useDifferentColors and self.data_has_discrete_class:
            color_discrete = True
            color_index = 3

        colors = []
        if color_discrete:
            for i in range(len(self.data_domain.classVar.values)):
                c = self.discPalette[i]
                colors.append([c.red()/255., c.green()/255., c.blue()/255.])

        self.set_shown_attributes_indices(0, 1, 2, color_index, symbol_index, size_index, label_index,
                                          colors, num_symbols_used,
                                          x_discrete, y_discrete, z_discrete,
                                          self.jitter_size, self.jitter_continuous,
                                          numpy.array([1., 1., 1.]), numpy.array([0., 0., 0.]))

        def before_draw_callback():
            # Override modelview (scatterplot points camera somewhat below the center, which doesn't
            # look good with sphere)
            modelview = QMatrix4x4()
            if self.camera_in_center:
                modelview.lookAt(
                    QVector3D(0, 0, 0),
                    QVector3D(self.camera[0]*self.camera_distance,
                              self.camera[1]*self.camera_distance,
                              self.camera[2]*self.camera_distance),
                    QVector3D(0, 1, 0))
                projection = QMatrix4x4()
                projection.perspective(90., float(self.width()) / self.height(),
                                       self.perspective_near, self.perspective_far)
                self.projection = projection
            else:
                modelview.lookAt(
                    QVector3D(self.camera[0]*self.camera_distance,
                              self.camera[1]*self.camera_distance,
                              self.camera[2]*self.camera_distance),
                    QVector3D(0, 0, 0),
                    QVector3D(0, 1, 0))
            self.modelview = modelview
            self.draw_sphere()

        self.before_draw_callback = before_draw_callback
        self.updateGL()

    def updateGraph(self, attrList=None, setAnchors=0, insideColors=None, **args):
        print('updateGraph')

    def draw_sphere(self):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.sphere_shader.bind()
        self.sphere_shader.setUniformValue('win_scale', self.width(), self.height())
        self.sphere_shader.setUniformValue('projection', self.projection)
        self.sphere_shader.setUniformValue('modelview', self.modelview)
        self.sphere_shader.setUniformValue('cam_position', QVector3D(*self.camera)*self.camera_distance)
        self.sphere_shader.setUniformValue('invert_transparency', self.camera_in_center)
        glBindVertexArray(self.sphere_vao_id)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        glDrawArrays(GL_TRIANGLES, 0, self.sphere_vao_id.num_vertices)
        glCullFace(GL_BACK)
        glDrawArrays(GL_TRIANGLES, 0, self.sphere_vao_id.num_vertices)
        glDisable(GL_CULL_FACE)

        glBindVertexArray(0)
        self.sphere_shader.release()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.projection.data(), dtype=float))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.modelview.data(), dtype=float))

        # Two more ways to visualize sphere:
        #glColor3f(0., 0., 0.)
        #glDisable(GL_BLEND)
        #num_rings = 24
        #num_parts = 30
        #r = 1.
        #glBegin(GL_LINES)
        #for i in range(num_rings):
            #z_offset = float(i) * 2 / num_rings - 1.
            #r = (1. - z_offset**2)**0.5
            #for j in range(num_parts):
                #angle_z_0 = float(j) * 2 * pi / num_parts
                #angle_z_1 = float(j+1) * 2 * pi / num_parts
                #glVertex3f(sin(angle_z_0)*r, cos(angle_z_0)*r, z_offset)
                #glVertex3f(sin(angle_z_1)*r, cos(angle_z_1)*r, z_offset)
            #for j in range(num_parts):
                #angle_z_0 = float(j) * 2 * pi / num_parts
                #angle_z_1 = float(j+1) * 2 * pi / num_parts
                #glVertex3f(sin(angle_z_0)*r, z_offset, cos(angle_z_0)*r)
                #glVertex3f(sin(angle_z_1)*r, z_offset, cos(angle_z_1)*r)
        #glEnd()

        #for i in range(num_rings):
            #angle_y = float(i) * 2 * pi / num_rings
            #glBegin(GL_LINES)
            #for j in range(num_parts):
                #angle_x_0 = float(j) * 2 * pi / num_parts
                #angle_x_1 = float(j+1) * 2 * pi / num_parts
                #glVertex3f(r * sin(angle_x_0) * cos(angle_y),
                           #r * cos(angle_x_0),
                           #r * sin(angle_x_0) * sin(angle_y))
                #glVertex3f(r * sin(angle_x_1) * cos(angle_y),
                           #r * cos(angle_x_1),
                           #r * sin(angle_x_1) * sin(angle_y))
            #glEnd()

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        radius = 1.

        if self.showAnchors:
            for anchor in self.anchor_data:
                x, y, z, label = anchor

                direction = QVector3D(x, y, z)
                up = QVector3D(0, 1, 0)
                right = QVector3D.crossProduct(direction, up).normalized()
                up = QVector3D.crossProduct(right, direction)
                rotation = QMatrix4x4()
                rotation.setColumn(0, QVector4D(right, 0))
                rotation.setColumn(1, QVector4D(up, 0))
                rotation.setColumn(2, QVector4D(direction, 0))

                self.cone_shader.bind()
                self.cone_shader.setUniformValue('projection', self.projection)
                modelview = QMatrix4x4(self.modelview)
                modelview.translate(x, y, z)
                modelview = modelview * rotation
                modelview.rotate(-90, 1, 0, 0)
                modelview.translate(0, -0.03, 0)
                modelview.scale(-0.03, -0.03, -0.03)
                self.cone_shader.setUniformValue('modelview', modelview)

                glBindVertexArray(self.cone_vao_id)
                glDrawArrays(GL_TRIANGLES, 0, self.cone_vao_id.num_vertices)
                glBindVertexArray(0)

                self.cone_shader.release()

                glColor4f(0, 0, 0, 1)
                self.renderText(x*1.2*radius, y*1.2*radius, z*1.2*radius, label)

            num_parts = 30
            anchors = array([a[:3] for a in self.anchor_data])
            for anchor in self.anchor_data:
                a0 = array(anchor[:3])
                neighbours = anchors.copy()
                neighbours = [(((n-a0)**2).sum(), n)  for n in neighbours]
                neighbours.sort(key=lambda e: e[0])
                for i in range(1, min(len(anchors), 4)): 
                    difference = neighbours[i][1]-a0
                    glBegin(GL_LINES)
                    for j in range(num_parts):
                        glVertex3f(*normalize(a0 + difference*j/float(num_parts)))
                        glVertex3f(*normalize(a0 + difference*(j+1)/float(num_parts)))
                    glEnd(GL_LINES)

    def setCanvasColor(self, c):
        pass

    def color(self, role, group=None):
        if group:
            return self.palette().color(group, role)
        else:
            return self.palette().color(role)

    def set_palette(self, palette):
        self.updateGL()

    def getSelectionsAsExampleTables(self, attrList, useAnchorData=1, addProjectedPositions=0):
        return (None, None)

    def removeAllSelections(self):
        pass

    def update_point_size(self):
        self.symbol_scale = self.point_width*self._point_width_to_symbol_scale
        self.updateGL()

    def update_alpha_value(self):
        self.updateGL()

    def replot(self):
        pass

    def mouseMoveEvent(self, event):
        self.invert_mouse_x = self.camera_in_center
        OWPlot3D.mouseMoveEvent(self, event)

class OWSphereviz3D(OWLinProjQt):
    settingsList = ['showAllAttributes']

    def __init__(self, parent=None, signalManager=None):
        OWLinProjQt.__init__(self, parent, signalManager, "Sphereviz 3D", graphClass=OWSphereviz3DPlot)

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
    viz = OWSphereviz3D()
    viz.show()
    data = orange.ExampleTable('../../doc/datasets/iris')
    viz.setData(data)
    viz.handleNewSignals()
    app.exec_()
