'''
Created on Aug 16, 2009

@author: Nejc Skofic
@contact: nejc.skofic@gmail.com
'''

import orange, numpy
from numpy import sqrt, abs, dot, transpose
from numpy.linalg import eig, inv

mathlib_import = True
try:
    import matplotlib.pyplot as plt
except:
    print "Warning: importing of matplotlib has failed.\nPlotting functions will be unavailable."
    mathlib_import = False
    
#color table for biplot
Colors = ['bo','go','yo','co','mo']

def defaultImputer(dataset):
    """Default imputer with average data imputaton."""
    return orange.ImputerConstructor_average(dataset)

def defaultContinuizer(dataset):
    """Default continuizer with:
        
    - multinomial -> as normalized ordinal
    - class -> ignore
    - continuous -> leave
        
    """
    continuizer = orange.DomainContinuizer()
    continuizer.multinomialTreatment = continuizer.AsNormalizedOrdinal
    continuizer.classTreatment = continuizer.Ignore
    continuizer.continuousTreatment = continuizer.Leave
    return continuizer(dataset)

def centerData(dataMatrix):
    """Perfomrs centering od data along rows, returns center and centered data."""
    center = numpy.sum(dataMatrix, axis = 0)/float(len(dataMatrix))
    return center, (dataMatrix - center)

def standardizeData(dataMatrix):
    """Performs standardization of data along rows. Throws error if constant
    variable is present."""
    scale = numpy.std(dataMatrix, axis = 0)
    if 0. in scale:
        raise orange.KernelException, "Constant variable, cannot standardize!"
    return scale, dataMatrix * 1./scale

class PCA(object):
    """Class that creates PCA projection
    
    Constructor parameters
    ----------------------
    dataset : orange.ExampleTable object. 
            If not present (default), constructor will just set all parameters
            needed for projection. Actual projection can then be performed with
            *pca(dataset)*
    attributes : list-like object containing names of attributes retained in
            PCA. If None (default) all attributes are retained
    rows : list-like object containg 0s and 1s for which rows should be retained.
            If None (default) all rows are retained
    standardize : True or False (default). If True performs standardisation of
            dataset
    imputer : orange.Imputer object. Defines how data is imputed if values are
            missing. Must NOT be trained. Default is average imputation
    continuizer : orange.Continuizer object. Defines how data is continuized if
            there are descrete attributes. Default:
            
                - multinomial -> as normalized ordinal
                - class -> ignore
                - continuous -> leave
                
    maxNumberOfComponents : How many components should be retained. Default is 10,
            if -1 all components will be retained
    varianceCovered : How much variance of data should be covered. Default is 0.95
    useGeneralizedVectors : True or False (default). If True, generalized
            eigenvectors are used
            
    Returns
    -------
    PCA object if dataset was None
    
    PCAClassifier object if projection was successful or None if projection has failed
    """
    
    def __new__(cls, dataset = None, attributes = None, rows = None, standardize = 0,
                 imputer = defaultImputer, continuizer = defaultContinuizer,
                 maxNumberOfComponents = 10, varianceCovered = 0.95,
                 useGeneralizedVectors = 0):
        
        learner = object.__new__(cls, {})
        
        if dataset:
            learner.__init__(attributes, rows, standardize, imputer, continuizer,
                             maxNumberOfComponents, varianceCovered, useGeneralizedVectors)
            return learner(dataset)
        else:
            return learner
    
    #decide what should be inmutable and what not
    #imutable for sure: imputer and continuizer
    def __init__(self, attributes = None, rows = None, standardize = 0,
                 imputer = defaultImputer, continuizer = defaultContinuizer,
                 maxNumberOfComponents = 10, varianceCovered = 0.95,
                 useGeneralizedVectors = 0):
        
        self.attributes = attributes
        self.rows = rows
        self.standardize = standardize
        self.imputer = imputer
        self.continuizer = continuizer
        self.maxNumberOfComponents = maxNumberOfComponents
        self.varianceCovered = varianceCovered
        self.useGeneralizedVectors = useGeneralizedVectors
    
    def __call__(self, dataset):
        
        #Modify dataset
        dataset = self._datasetPreprocessing(dataset)
        dataMatrix, classArray, center, deviation = self._dataMatrixPreprocessing(dataset)
        
        #Perform projection
        evalues, evectors = self._createPCAProjection(dataMatrix, classArray)
        
        #check if return status is None, None
        if (evalues, evectors) == (None, None):
            print "Principal component could not be performed (complex eigenvalues or singular matrix if generalized eigenvectors were used)"
            return None
        
        #return PCAClassifier
        return PCAClassifier(domain = self.attributes,
                             imputer = self.imputer,
                             continuizer = self.continuizer,
                             center = center,
                             deviation = deviation,
                             evalues = evalues,
                             loadings = evectors)

    def _datasetPreprocessing(self, dataset):
        """
        First remove unwanted attributes, save domain (so that PCA remembers on
        what kind of dataset it was trained), remove unwanted rows and impute
        values and continuize.
        """

        if self.attributes:
            dataset = dataset.select(self.attributes)
        
        #we need to retain only selected attributes without class attribute    
        self.attributes = [att.name for att in dataset.domain.attributes]

        imputer = self.imputer(dataset)
            
        if self.rows:
            dataset = dataset.select(self.rows)


        dataset = imputer(dataset)
        domain = self.continuizer(dataset)
        dataset = dataset.translate(domain)

        return dataset        

    def _dataMatrixPreprocessing(self, dataset):
        """
        Creates numpy arrays, center dataMatrix and standardize it if that
        option was selected, and return dataMatrix, classArray, center and
        deviation.
        """

        dataMatrix, classArray, x = dataset.toNumpy()
        
        center, dataMatrix = centerData(dataMatrix)
        
        deviation = None
        if self.standardize:
            deviation, dataMatrix = standardizeData(dataMatrix)

        return dataMatrix, classArray, center, deviation
    
    def _createPCAProjection(self, dataMatrix, classArray):
        """
        L -> Laplacian weight matrix constructed from classArray or identity if
             classArray is None
        M -> dataMatrix
        
        Normal method: t(M) * L * M * x = lambda * x
        Snapshot method: M * t(M) * L * M * x = lambda * M * x
        Generalized vectors: (t(M) * M)^(-1) * t(M) * L * M * x = lambda * x
        Snapshot with generalized vectors: M * (t(M) * M)^(-1) * t(M) * L * M * x = lambda * M * x
        """
        
        n, d = numpy.shape(dataMatrix)
        
        if classArray != None:
            L = numpy.zeros((len(dataMatrix), len(dataMatrix)))
            for i in range(len(dataMatrix)):
                for j in range(i+1, len(dataMatrix)):
                    L[i,j] = -int(classArray[i] != classArray[j])
                    L[j,i] = -int(classArray[i] != classArray[j])

            s = numpy.sum(L, axis=0)      # doesn't matter which axis since the matrix L is symmetrical
            for i in range(len(dataMatrix)):
                L[i,i] = -s[i]
                
            matrix = dot(transpose(dataMatrix), L)
        else:
            matrix = transpose(dataMatrix)
        
        if self.useGeneralizedVectors:
            temp_matrix = dot(transpose(dataMatrix), dataMatrix)
            try:
                temp_matrix = inv(temp_matrix)
            except:
                return None, None
            matrix = dot(temp_matrix, matrix)
        
        if n < d:
            #snapshot method
            covMatrix = dot(dataMatrix, matrix)
            trace = numpy.trace(covMatrix)
            scale = n / trace
            evalues, evectors = eig(covMatrix)
            if evalues.dtype.kind == 'c':
                return None, None
            positiveArray = numpy.array([value > 0. for value in evalues])
            evalues = evalues[positiveArray]
            evectors = evectors[positiveArray]
            evectors = transpose(1./sqrt(evalues)) * transpose(dot(evectors, dataMatrix))
        else:
            covMatrix = dot(matrix, dataMatrix)
            trace = numpy.trace(covMatrix)
            scale = d / trace
            evalues, evectors = eig(covMatrix)
            if evalues.dtype.kind == 'c':
                return None, None
            positiveArray = numpy.array([value > 0. for value in evalues])
            evalues = evalues[positiveArray]
            evectors = evectors[positiveArray]
                        
        order = (numpy.argsort(evalues)[::-1])
        N = len(evalues)
        maxComp = self.maxNumberOfComponents
        variance = self.varianceCovered
        
        if maxComp == -1:
            maxComp = N
        maxComp = min(maxComp, N)

        order = order[:maxComp]
        evalues = numpy.take(evalues, order)

        #evalues are scaled -> value > 1 : explains more than previous attribute
        #                   -> value < 1 : explains less than previous attribute
        for i in range(len(order)):
            variance -= evalues[i] / trace 
            if variance < 0:
                return evalues[: i + 1] * scale, numpy.take(evectors, order[: i + 1], 1)
        
        return evalues * scale, numpy.take(evectors, order, 1) 

class PCAClassifier(object):
    
    def __init__(self, domain, imputer, continuizer, center, deviation, evalues, loadings):
        #data checking and modifying
        self.domain = domain
        self.imputer = imputer
        self.continuizer = continuizer
        #PCA properites
        self.center = center
        self.deviation = deviation
        self.evalues = evalues
        self.loadings = loadings
        
        #last predicition performed -> used for biplot
        self._dataMatrix = None
        self._classArray = None
    
    def __call__(self, dataset):
                
        try:
            #retain class attribute
            attrDataset = dataset.select(self.domain)
            imputer = self.imputer(attrDataset)
            attrDataset = imputer(attrDataset)
            domain = self.continuizer(attrDataset)
            attrDataset = attrDataset.translate(domain)
        except TypeError, e:
            raise orange.KernelException, "One or more attributes form training set are missing!"

        dataMatrix, classArray, x = attrDataset.toNumpy()

        dataMatrix -= self.center
        if self.deviation != None:
            dataMatrix *= 1./self.deviation
            
        #save transformed data
        self._dataMatrix = numpy.dot(dataMatrix, self.loadings)

        attributes = [orange.FloatVariable("PC%d" % (i + 1, )) for i in range(len(self.evalues))]
        new_domain = orange.Domain(attributes)
        new_table = orange.ExampleTable(new_domain, self._dataMatrix)

        if dataset.domain.classVar:
            #suboptimal
            classTable = dataset.select([dataset.domain.classVar.name])
            self._classArray = numpy.array([row.getclass() for row in classTable])
            new_table = orange.ExampleTable([new_table, classTable])
        
        return new_table
    
    def __str__(self):
        
        n, d = numpy.shape(self.loadings)
        comulative = 0.0
        
        summary = "PCA SUMMARY\n\nCenter:\n\n"
        summary += " %15s  " * len(self.domain) % tuple(self.domain) + "\n"
        summary += " %15.4f  " * len(self.center) % tuple(self.center) + "\n\n"
    
        if self.deviation != None:
            summary += "Deviation:\n\n"
            summary += " %15s  " * len(self.domain) % tuple(self.domain) + "\n"
            summary += " %15.4f  " * len(self.deviation) % tuple(self.deviation) + "\n\n"
   
        summary += "Importance of components:\n\n %12s   %12s   %12s\n" % ("eigenvalues", "proportion", "cumulative")
        for evalue in self.evalues:
            comulative += evalue/n
            summary += " %12.4f   %12.4f   %12.4f\n" % (evalue, evalue/n, comulative)
        
        summary += "\nLoadings:\n\n"
        summary += "      PC%d" * d % tuple(range(1, d + 1)) + "\n"
        for attr_num in range(len(self.domain)):
            summary += " % 8.4f" * d % tuple(self.loadings[attr_num])
            summary += "   %-30s\n" % self.domain[attr_num]

        return summary
    
    ################ Ploting functions ###################
    
    def plot(self, title = 'Scree plot', filename = 'scree_plot.png'):
        """
        Draws a scree plot. Matplotlib is needed.
        
        Parameters
        ----------
        title : Title of the plot
        filename : File name under which plot will be saved (default: scree_plot.png)
            If None, plot will be shown
        """
        
        if not mathlib_import:
            raise orange.KernelException, "Matplotlib was not imported!"
        
        #plt.clf() -> opens two windows
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        x_axis = range(len(self.evalues))
        x_labels = ["PC%d" % (i + 1, ) for i in x_axis]
        
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Principal components')
        ax.set_ylabel('Variance')
        ax.set_title(title + "\n")
        ax.bar(left = x_axis, height = self.evalues, align = 'center')
        ax.axis([-0.5, len(self.evalues) - 0.5, 0, self.evalues[0]*1.05])
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
    def biplot(self, choices = [1,2], scale = 1., title = 'Biplot',
               filename = 'biplot.png'):
        """
        Draws biplot for PCA. Matplotlib is needed. Actual projection must be
        performed via pca(data) before bilpot can be used.
        
        Parameters
        ----------
        choices : lenght 2 list-like object for choosing which 2 components
            should be used in biplot. Default is first and second
        scale : scale factor (default is 1.). Should be inside [0, 1]
        title : title of biplot
        filename : File name under which plot will be saved (default: biplot.png)
            If None, plot will be shown
        """
        
        if not mathlib_import:
            raise orange.KernelException, "Matplotlib was not imported!"
        
        if self._dataMatrix == None:
            raise orange.KernelException, "No data available for biplot!"
        
        if len(choices) != 2:
            raise orange.KernelException, 'You have to choose exactly two components'
        
        if max(choices[0], choices[1]) > len(self.evalues) or min(choices[0], choices[1]) < 1:
            raise orange.KernelException, 'Invalid choices'
        
        choice = numpy.array([i == choices[0] - 1 or i == choices[1] - 1 for i in range(len(self.evalues))])
        
        dataMatrix = self._dataMatrix[:,choice]
        loadings = self.loadings[:,choice]
        lam = (numpy.array(self.evalues)[choice])
        lam *= sqrt(len(self._dataMatrix))
        
        if scale < 0. or scale > 1.:
            print "Warning: 'scale' is outside [0, 1]"
        lam = lam**scale
        
        #TO DO -> pc.biplot (maybe)
        trDataMatrix = dataMatrix / lam
        trLoadings = loadings * lam
        
        max_data_value = numpy.max(abs(trDataMatrix)) * 1.05
        max_load_value = numpy.max(abs(trLoadings)) * 1.5
        
        #plt.clf()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title(title + "\n")
        ax1.set_xlabel("PC%s" % (choices[0]))
        ax1.set_ylabel("PC%s" % (choices[1]))
        ax1.xaxis.set_label_position('bottom')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_label_position('left')
        ax1.yaxis.set_ticks_position('left')        
        
        if self._classArray == None:
            trDataMatrix = transpose(trDataMatrix)
            ax1.plot(trDataMatrix[0], trDataMatrix[1], Colors[0])
        else:
            #suboptimal
            classValues = []
            for classValue in self._classArray:
                if classValue not in classValues:
                    classValues.append(classValue)
            for i in range(len(classValues)):
                choice = numpy.array([classValues[i] == cv for cv in self._classArray])
                partialDataMatrix = transpose(trDataMatrix[choice])
                ax1.plot(partialDataMatrix[0], partialDataMatrix[1],
                         Colors[i % len(Colors)], label = str(classValues[i]))
            ax1.legend()
        
        ax1.set_xlim(-max_data_value, max_data_value)
        ax1.set_ylim(-max_data_value, max_data_value)
        
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
        #using annotations instead of arrows because there is a strange implementation
        #of arrows in matplotlib version 0.99
        for i in range(len(trLoadings)):
            x, y = trLoadings[i]
            ax2.annotate('', (x, y), (0, 0), arrowprops = arrowprops)
            ax2.text(x * 1.1, y * 1.2, self.domain[i], color = 'red')
            
        ax2.set_xlim(-max_load_value, max_load_value)
        ax2.set_ylim(-max_load_value, max_load_value)
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()