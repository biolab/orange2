import Orange.data
import Orange.feature
import numpy as np
    
#color table for biplot
Colors = ['bo','go','yo','co','mo']

class Pca(object):
    """
    Orthogonal transformation of data into a set of uncorrelated variables called
    principal components. This transformation is defined in such a way that the
    first variable has as high variance as possible.

    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    :param standardize: Perform standardization of the dataset
    :type standardize: boolean
    :param max_components: Maximum number of retained components
    :type max_components: int
    :param variance_covered: Percent of the variance to cover with components
    :type variance_covered: float
        
    :rtype: :class:`Orange.projection.pca.Pca` or
            :class:`Orange.projection.pca.PcaClassifier`
    """
    
    def __new__(cls, dataset = None, **kwds):
        learner = object.__new__(cls)
        learner.__init__(**kwds)
        
        if dataset:
            return learner(dataset)
        else:
            return learner
    
    def __init__(self, standardize = True,
                 max_components = 0, variance_covered = 1):
        self.standardize = standardize
        self.max_components = max_components
        self.variance_covered = variance_covered if variance_covered < 1. else 1
    
    def __call__(self, dataset):
        """
        Perform a pca analysis on a dataset and return a classifier that maps data
        into principal component subspace.
        """
        
        X = dataset.to_numpy_MA("a")[0]
        N,M = X.shape
        Xm = np.mean(X, axis=0)
        Xd = X - Xm
        
        #take care of the constant features
        stdev = np.std(Xd, axis=0)
        relevant_features = stdev != 0
        if self.standardize:
            stdev[stdev == 0] = 1.
            Xd /= stdev
        Xd = Xd[:,relevant_features]
        
        #actual pca
        n,m = Xd.shape
        if n < m:
            C = np.ma.dot(Xd, Xd.T)
            V, D, T = np.linalg.svd(C)
            U = np.ma.dot(V.T, Xd) / np.sqrt(D.reshape(-1,1))
        else:
            C = np.ma.dot(Xd.T, Xd)
            U, D, T = np.linalg.svd(C)
        
        #insert zeros for constant features
        n, m = U.shape
        if m != M:
            U_ = np.zeros((n,M))
            U_[:,relevant_features] = U
            U = U_
        
        variance_sum = D.sum()
        
        #select eigen vectors
        if self.variance_covered != 1:
            nfeatures = np.nonzero(np.cumsum(D) / sum(D) >= self.variance_covered)[0][0] + 1
            U = U[:nfeatures, :]
            D = D[:nfeatures]
        
        if self.max_components > 0:
            U = U[:self.max_components, :]
            D = D[:self.max_components]
        
        n, m = U.shape
        pc_domain = Orange.data.Domain([Orange.feature.Continuous("Comp.%d"%(i+1)) for i in range(n)], False)
        
        return PcaClassifier(input_domain = dataset.domain,
                             pc_domain = pc_domain,
                             mean = Xm,
                             stdev = stdev,
                             standardize = self.standardize,
                             eigen_vectors = U,
                             eigen_values = D,
                             variance_sum = variance_sum)


class PcaClassifier(object):
    """
    .. attribute:: input_domain
    
        Domain of the dataset that was used to construct principal component
        subspace.
        
    .. attribute:: pc_domain
    
        Domain used in returned datasets. This domain has a Float variable for
        each principal component and no class variable.
        
    .. attribute:: mean
    
        Array containing means of each variable in the dataset that was used
        to construct pca space.
        
    .. attribute:: stdev
    
        An array containing standard deviations of each variable in the dataset
        that was used to construct pca space.
        
    .. attribute:: standardize
    
        True, if standardization was used when constructing the pca space. If set,
        instances will be standardized before being mapped to the pca space.
    
    .. attribute:: eigen_vectors
    
        Array containing vectors that are used to map to pca space.
        
    .. attribute:: eigen_values
    
        Array containing standard deviations of principal components.
    
    .. attribute:: variance_sum
    
        Sum of all variances in the dataset that was used to construct the pca
        space.
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    
    def __call__(self, dataset):
        if type(dataset) != Orange.data.Table:
            dataset = Orange.data.Table([dataset])

        X = dataset.to_numpy_MA("a")[0]
        Xm, U = self.mean, self.eigen_vectors
        n, m = X.shape
        
        if m != len(self.stdev):
            raise orange.KernelException, "Invalid number of features"
        
        Xd = X - Xm
        
        if self.standardize:
            Xd /= self.stdev
        
        self.A = np.ma.dot(Xd, U.T)
        
        return Orange.data.Table(self.pc_domain, self.A.tolist())
    
    def __str__(self):
        ncomponents = 10
        s = self.variance_sum
        cs = np.cumsum(self.eigen_values) / s
        return "\n".join([
        "PCA SUMMARY",
        "",
        "Std. deviation of components:",
        " ".join(["              "] +
                 ["%10s" % a.name for a in self.pc_domain.attributes]),
        " ".join(["Std. deviation"] +
                 ["%10.3f" % a for a in self.eigen_values]),
        " ".join(["Proportion Var"] + 
                 ["%10.3f" % a for a in  self.eigen_values / s * 100]),
        " ".join(["Cumulative Var"] +
                 ["%10.3f" % a for a in cs * 100]),
        "",
        #"Loadings:",
        #" ".join(["%10s"%""] + ["%10s" % a.name for a in self.pc_domain]),
        #"\n".join([
        #    " ".join([a.name] + ["%10.3f" % b for b in self.eigen_vectors.T[i]])
        #          for i, a in enumerate(self.input_domain.attributes)
        #          ])
        ]) if len(self.pc_domain) <= ncomponents else \
        "\n".join([
        "PCA SUMMARY",
        "",
        "Std. deviation of components:",
        " ".join(["              "] +
                 ["%10s" % a.name for a in self.pc_domain.attributes[:ncomponents]] +
                 ["%10s" % "..."] +
                 ["%10s" % self.pc_domain.attributes[-1].name]),
        " ".join(["Std. deviation"] +
                 ["%10.3f" % a for a in self.eigen_values[:ncomponents]] + 
                 ["%10s" % ""] +
                 ["%10.3f" % self.eigen_values[-1]]),
        " ".join(["Proportion Var"] + 
                 ["%10.3f" % a for a in self.eigen_values[:ncomponents] / s * 100] + 
                 ["%10s" % ""] +
                 ["%10.3f" % (self.eigen_values[-1] / s * 100)]),
        " ".join(["Cumulative Var"] +
                 ["%10.3f" % a for a in cs[:ncomponents] * 100] + 
                 ["%10s" % ""] +
                 ["%10.3f" % (cs[-1] * 100)]),
        "",
        #"Loadings:",
        #" ".join(["%16s" % ""] +
        #         ["%8s" % a.name for a in self.pc_domain.attributes[:ncomponents]] +
        #         ["%8s" % "..."] +
        #         ["%8s" % self.pc_domain.attributes[-1].name]),
        #"\n".join([
        #    " ".join(["%16.16s" %a.name] +
        #             ["%8.3f" % b for b in self.eigen_vectors.T[i, :ncomponents]] +
        #             ["%8s" % ""] +
        #             ["%8.3f" % self.eigen_vectors.T[i, -1]])
        #          for i, a in enumerate(self.input_domain.attributes)
        #          ])
        ])

        
    
    ################ Plotting functions ###################
    
    def scree_plot(self, filename = None, title = 'Scree plot'):
        """
        Draw a scree plot of principal components
        
        :param filename: Name of the file to which the plot will be saved. \
        If None, plot will be displayed instead.
        :type filename: str
        :param title: Plot title
        :type title: str
        """
        import pylab as plt
        
        s = self.variance_sum
        vc = self.eigen_values / s
        cs = np.cumsum(self.eigen_values) / s
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        x_axis = range(len(self.eigen_values))
        x_labels = ["PC%d" % (i + 1, ) for i in x_axis]
        
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_labels)
        plt.setp(ax.get_xticklabels(), "rotation", 90)
        plt.grid(True)
        
        ax.set_xlabel('Principal components')
        ax.set_ylabel('Proportion of Variance')
        ax.set_title(title + "\n")
        ax.plot(x_axis, vc, color="red")
        ax.scatter(x_axis, vc, color="red", label="Variance")
        
        ax.plot(x_axis, cs, color="orange")
        ax.scatter(x_axis, cs, color="orange", label="Cumulative Variance")
        ax.legend(loc=0)
        
        ax.axis([-0.5, len(self.eigen_values) - 0.5, 0, 1])
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
    def biplot(self, filename = None, components = [0,1], title = 'Biplot'):
        """
        Draw biplot for PCA. Actual projection must be performed via pca(data)
        before bipot can be used.
        
        :param filename: Name of the file to which the plot will be saved. \
        If None, plot will be displayed instead.
        :type plot: str
        :param components: List of two components to plot.
        :type components: list
        :param title: Plot title
        :type title: str
        """
        import pylab as plt
        
        if len(components) < 2:
            raise orange.KernelException, 'Two components are needed for biplot'
        
        if not (0 <= min(components) <= max(components) < len(self.eigen_values)):
            raise orange.KernelException, 'Invalid components'
        
        X = self.A[:,components[0]]
        Y = self.A[:,components[1]]
        
        vectorsX = self.eigen_vectors[:,components[0]]
        vectorsY = self.eigen_vectors[:,components[1]]
        
        
        #TO DO -> pc.biplot (maybe)
        #trDataMatrix = dataMatrix / lam
        #trLoadings = loadings * lam
        
        #max_data_value = numpy.max(abs(trDataMatrix)) * 1.05
        max_load_value = self.eigen_vectors.max() * 1.5
        
        #plt.clf()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title(title + "\n")
        ax1.set_xlabel("PC%s (%d%%)" % (components[0], self.eigen_values[components[0]] / self.variance_sum * 100))
        ax1.set_ylabel("PC%s (%d%%)" % (components[1], self.eigen_values[components[1]] / self.variance_sum * 100))
        ax1.xaxis.set_label_position('bottom')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_label_position('left')
        ax1.yaxis.set_ticks_position('left')        
        
        #if self._classArray == None:
        #trDataMatrix = transpose(trDataMatrix)
        ax1.plot(X, Y, Colors[0])
        #else:
            #suboptimal
        #    classValues = []
        #    for classValue in self._classArray:
        #        if classValue not in classValues:
        #            classValues.append(classValue)
        #    for i in range(len(classValues)):
        #        choice = numpy.array([classValues[i] == cv for cv in self._classArray])
        #        partialDataMatrix = transpose(trDataMatrix[choice])
        #        ax1.plot(partialDataMatrix[0], partialDataMatrix[1],
        #                 Colors[i % len(Colors)], label = str(classValues[i]))
        #    ax1.legend()
        
        #ax1.set_xlim(-max_data_value, max_data_value)
        #ax1.set_ylim(-max_data_value, max_data_value)
        
        #eliminate double axis on right
        ax0 = ax1.twinx()
        ax0.yaxis.set_visible(False)
                
        ax2 = ax0.twiny()
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.set_ticks_position('top')
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_ticks_position('right')
        for tl in ax2.get_xticklabels():
            tl.set_color('r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        
        arrowprops = dict(facecolor = 'red', edgecolor = 'red', width = 1, headwidth = 4)

        for (x, y, a) in zip(vectorsX, vectorsY,self.input_domain.attributes):
            if max(x, y) < 0.1:
                continue
            print x, y, a
            ax2.annotate('', (x, y), (0, 0), arrowprops = arrowprops)
            ax2.text(x * 1.1, y * 1.2, a.name, color = 'red')
            
        ax2.set_xlim(-max_load_value, max_load_value)
        ax2.set_ylim(-max_load_value, max_load_value)
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
 
