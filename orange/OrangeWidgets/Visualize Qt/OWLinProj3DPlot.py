from plot.owplot3d import *
from plot.owplotgui import *

from Orange.preprocess.scaling import ScaleLinProjData3D
import orange
Discrete = orange.VarTypes.Discrete
Continuous = orange.VarTypes.Continuous

class OWLinProj3DPlot(OWPlot3D, ScaleLinProjData3D):
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

        self.gui = OWPlotGUI(self)

    def setData(self, data, subsetData=None, **args):
        ScaleLinProjData3D.setData(self, data, subsetData, **args)
        #self.initializeGL() # Apparently this is not called already
        self.makeCurrent()

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

        self.updateGL()

    def updateGraph(self, attrList=None, setAnchors=0, insideColors=None, **args):
        print('updateGraph')

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
