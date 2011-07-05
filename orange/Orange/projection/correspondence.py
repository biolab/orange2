"""\
********************************************
Correspondence Analysis (``correspondence``)
********************************************

Correspondence analysis is a descriptive/exploratory technique designed
to analyze simple two-way and multi-way tables containing some measure
of correspondence between the rows and columns. It does this by mapping
the rows and columns into two sets of factor scores respectively while
preserving the similarity structure of the table's rows and columns.

It is similar in nature to PCA but is used for analysis of
quantitative data.

.. autoclass:: CA
    :members:
    :exclude-members: A, B, D, F, G
             


Example
-------

Data table given below represents smoking habits of different employees
in a company (computed from smokers_ct.tab).

    ================  ====  =====  ======  =====  ==========
                           Smoking category
    ----------------  --------------------------  ----------
    Staff group       None  Light  Medium  Heavy  Row totals
    ================  ====  =====  ======  =====  ==========
    Senior managers   4     2      3       2      11
    Junior managers   4     3      7       4      18
    Senior employees  25    10     12      2      51
    Junior employees  18    24     33      12     88
    Secretaries       10    6      7       2      25
    Column totals     61    45     62      25     193
    ================  ====  =====  ======  =====  ==========

The 4 column values in each row of the table can be viewed as coordinates
in a 4-dimensional space, and the (Euclidean) distances could be computed
between the 5 row points in the 4-dimensional space. The distances
between the points in the 4-dimensional space summarize all information
about the similarities between the rows in the table above.
Correspondence analysis module can be used to find a
lower-dimensional space, in which the row points are positioned in a
manner that retains all, or almost all, of the information about the
differences between the rows. All information about the similarities
between the rows (types of employees in this case) can be presented in a
simple 2-dimensional graph. While this may not appear to be particularly
useful for small tables like the one shown above, the presentation and
interpretation of very large tables (e.g., differential preference for
10 consumer items among 100 groups of respondents in a consumer survey)
could greatly benefit from the simplification that can be achieved via
correspondence analysis (e.g., represent the 10 consumer items in a
2-dimensional space). This analysis can be similarly performed on columns
of the table.

So lets load the data, compute the contingency and do the analysis
(`correspondence.py`_, uses `smokers_ct.tab`_))::
    
    from Orange.projection import correspondence
    from Orange.statistics import contingency
    
    data = Orange.data.Table("smokers_ct.tab")
    staff = data.domain["Staff group"]
    smoking = data.domain["Smoking category"]
    
    # Compute the contingency
    cont = contingency.VarVar(staff, smoking, data) 
    
    c = correspondence.CA(cont, staff.values, smoking.values)
    
    print "Row profiles"
    print c.row_profiles()
    print 
    print "Column profiles"
    print c.column_profiles()
    
which produces matrices of relative frequencies (normalized across rows
and columns respectively) ::
    
    Column profiles:
    [[ 0.06557377  0.06557377  0.40983607  0.29508197  0.16393443]
     [ 0.04444444  0.06666667  0.22222222  0.53333333  0.13333333]
     [ 0.0483871   0.11290323  0.19354839  0.53225806  0.11290323]
     [ 0.08        0.16        0.16        0.52        0.08      ]]
    
    Row profiles:
    [[ 0.36363636  0.18181818  0.27272727  0.18181818]
     [ 0.22222222  0.16666667  0.38888889  0.22222222]
     [ 0.49019608  0.19607843  0.23529412  0.07843137]
     [ 0.20454545  0.27272727  0.375       0.14772727]
     [ 0.4         0.24        0.28        0.08      ]]
    
The points in the two-dimensional correspondence analysis display that are
close to each other are similar with regard to the pattern of relative
frequencies across the columns, i.e. they have similar row profiles.
After producing the plot it can be noticed that along the most important
first axis in the plot, the Senior employees and Secretaries are relatively
close together. This can be also seen by examining row profile, these two
groups of employees show very similar patterns of relative frequencies
across the categories of smoking intensity. ::

    c.plot_biplot()
    
.. image:: files/correspondence-biplot.png

Lines 26-29 print out singular values , eigenvalues, percentages of
inertia explained. These are important values to decide how many axes
are needed to represent the data. The dimensions are "extracted" to
maximize the distances between the row or column points, and successive
dimensions will "explain" less and less of the overall inertia. ::
    
    print "Singular values: " + str(diag(c.D))
    print "Eigen values: " + str(square(diag(c.D)))
    print "Percentage of Inertia:" + str(c.inertia_of_axes() / sum(c.inertia_of_axes()) * 100.0)
    print
    
which outputs::
    
    Singular values: 
    [  2.73421115e-01   1.00085866e-01   2.03365208e-02   1.20036007e-16]
    Eigen values: 
    [  7.47591059e-02   1.00171805e-02   4.13574080e-04   1.44086430e-32]
    Percentage of Inertia:
    [  8.78492893e+01   1.16387938e+01   5.11916964e-01   1.78671526e-29]

Lines 31-35 print out principal row coordinates with respect to first
two axes. And lines 24-25 show decomposition of inertia. ::

    print "Principal row coordinates:"
    print c.row_factors()
    print 
    print "Decomposition Of Inertia:"
    print c.column_inertia()
    
Lets also plot the scree diagram. Scree diagram is a plot of the
amount of inertia accounted for by successive dimensions, i.e.
it is a plot of the percentage of inertia against the components,
plotted in order of magnitude from largest to smallest. This plot is
usually used to identify components with the highest contribution of
inertia, which are selected, and then look for a change in slope in
the diagram, where the remaining factors seem simply to be debris at
the bottom of the slope and they are discarded. ::

    c.plot_scree_diagram()
    
.. image:: files/correspondence-scree-plot.png

Multi-Correspondence Analysis
=============================



    
Utility Functions
-----------------

.. autofunction:: burt_table

.. _correspondence.py: code/correspondence.py
.. _smokers_ct.tab: code/smokers_ct.tab

"""

import numpy
import numpy.linalg
import orange
import operator

from Orange.misc import deprecation_warning
from Orange.statistics import contingency


def input(filename):
    """ Loads contingency matrix from the file. The expected format is a
    simple tab delimited matrix of numerical values.
    
    :param filename: Filename
    :type filename: str
    
    """
    table = [[],[]]
    try:
        file = open(filename)
        try:
            table = file.readlines()
        finally:
            file.close()
        table = [map(int, el.split()) for el in table]
    except IOError:
        pass
    return table
    

class CA(object):
    """ Main class used for computation of correspondence analysis.
    """
    def __init__(self, contingency_table, row_labels = [], col_labels= []):
        """ Initialize the CA instance
        
        :param contingency_table: A contingency table (can be
            :class:`Orange.statistics.contingency.Table` or
            :class:`numpy.ndarray` or list-of-lists)
        
        :param row_labels: An optional list of row labels
        :type row_labels: list
        :param col_labels: An optional list of column labels
        :type col_labels: list
        
        """     
        # calculating correspondence analysis from the data matrix
        # algorithm described in the book (put reference) is used
        
        self.row_labels = row_labels
        self.col_labels = col_labels
        
        if isinstance(contingency_table, contingency.Table):
            contingency_table = numpy.asarray([[e for e in row] for row in contingency_table])
            
        self.__data_matrix = numpy.matrix(contingency_table, float)
        sum_elem = numpy.sum(self.__data_matrix)
        
        # __corr is a matrix of relative frequencies of elements in data matrix
        self.__corr = self.__data_matrix / sum_elem
        self.__col_sums = numpy.sum(self.__corr, 0)
        self.__row_sums = numpy.sum(self.__corr, 1)
        
        self.__col_profiles = numpy.matrix(numpy.diag((1. / numpy.array(self.__col_sums))[0])) * self.__corr.T
        self.__row_profiles = numpy.matrix(numpy.diag((1. / numpy.array(self.__row_sums).reshape(1, -1))[0])) * self.__corr
        
        self.__a, self.__d, self.__b = self.__calculate_svd();
        
        self.__f = numpy.diag((1. / self.__row_sums).reshape(1,-1).tolist()[0]) * self.__a * self.__d
        self.__g = numpy.diag((1. / self.__col_sums).tolist()[0]) * self.__b * self.__d.T
        
    def __calculate_svd(self):
        """ Computes generalized SVD ...
            
        This function is used to calculate decomposition A = N * D_mi * M' , where N' * diag(row_sums) * N = I and
        M' * diag(col_sums) * M = I. This decomposition is calculated in 4 steps:
        
            i)   B = diag(row_sums)^1/2 * A * diag(col_sums)^1/2
            ii)  find the ordinary SVD of B: B = U * D * V'
            iii) N = diag(row_sums)^-1/2 * U
                 M = diag(col_sums)^-1/2 * V
                 D_mi = D
            iv)  A = N * D_mi * M'
        
        Returns (N, D_mi, M)
                    
        """
        
        a = self.__corr - self.__row_sums * self.__col_sums
        b = numpy.diag(numpy.sqrt((1. / self.__row_sums).reshape(1,-1).tolist()[0])) * a * numpy.diag(numpy.sqrt((1. / self.__col_sums).tolist()[0]))
        u, d, v = numpy.linalg.svd(b, 0)
        N = numpy.diag(numpy.sqrt(self.__row_sums.reshape(1, -1).tolist()[0])) * u
        M = numpy.diag(numpy.sqrt(self.__col_sums.tolist()[0])) * numpy.transpose(v)
        d = numpy.diag(d.tolist())
        
        return (N, d, M)       
        
    @property
    def data_matrix(self):
        """ The :obj:`numpy.matrix` object that is representation of
        input contingency table.
        """
        return self.__data_matrix   

    @property
    def A(self): 
        """ columns of A define the principal axes of the column clouds
        (same as row_principal_axes)
        """
        return self.__a
    
    @property
    def D(self): 
        """ elements on diagonal of D are singular values.
        """
        return self.__d
    
    @property
    def B(self): 
        """ columns of B defines the principal axes of the row clouds
        (same as column_principal_axes)
        """        
        return self.__b
    
    @property
    def F(self): 
        """ coordinates of the row profiles with respect to principal axes B
        (same as row_factors).
        """
        return self.__f
    
    @property
    def G(self): 
        """ coordinates of the column profiles with respect to principal axes A
        (same as column_factors).
        """
        return self.__g
    
    @property
    def row_principal_axes(self):
        """ A :obj:`numpy.matrix` of principal axes (in columns)
        of the row points.
         
        """
        return self.__b
    
    @property
    def column_principal_axes(self):
        """ A :obj:`numpy.matrix` of principal axes (in columns)
        of the column points. 
        """
        return self.__b
    
    def row_factors(self):
        """ Return a :obj:`numpy.matrix` of factor scores (coordinates)
        for rows of the input matrix.        
        """
        return self.__f
    
    def column_factors(self):
        """ Return a :obj:`numpy.matrix` of factor scores (coordinates)
        for columns of the input matrix.
        """ 
        return self.__g
    
    def row_profiles(self):
        """ Return a :obj:`numpy.matrix` of row profiles, i.e. rows of the
        ``data matrix`` normalized by the row sums.
        """
        return self.__row_profiles
    
    def column_profiles(self):
        """ Return a :obj:`numpy.matrix` of column profiles, i.e. rows of the
        ``data_matrix`` normalized by the column sums.
        """
        return self.__row_profiles
        
    def row_inertia(self):
        """ Return the contribution of rows to the inertia across principal axes. 
        """
        return numpy.multiply(self.__row_sums, numpy.multiply(self.__f, self.__f))
    
    def column_inertia(self):
        """ Return the contribution of columns to the inertia across principal axes. 
        """
        return numpy.multiply(numpy.transpose(self.__col_sums), numpy.multiply(self.__g, self.__g))
        
    def inertia_of_axes(self):
        """ Return :obj:`numpy.ndarray` whose elements are inertias of principal axes. 
        """
        return numpy.array(numpy.sum(self.column_inertia(), 0).tolist()[0])
    
    def ordered_row_indices(self, axes=None):
        """ Return indices of rows with most inertia. If ``axes`` is given
        take only inertia in those those principal axes into account.
        
        :param axes: Axes to take into account.
        """
        inertia = self.row_inertia()
        if axes:
            inertia = inertia[:, axes]
        indices = numpy.ravel(numpy.argsort(numpy.sum(inertia, 1), 0))
        return list(reversed(indices))
    
    def ordered_column_indices(self, axes=None):
        """ Return indices of rows with most inertia. If ``axes`` is given
        take only inertia in those principal axes into account.
        
        :param axes: Axes to take into account.
        """
        inertia = self.column_inertia()
        if axes:
            inertia = inertia[:, axes]
        
        indices = numpy.ravel(numpy.argsort(numpy.sum(inertia, 1), 0))
        return list(reversed(indices))
    
    def plot_scree_diagram(self):
        """ Plot a scree diagram of the inertia.
        """
        import pylab
        inertia = self.inertia_of_axes()
        inertia *= 100.0 / numpy.sum(inertia)
        
        pylab.plot(range(1, min(self.__data_matrix.shape) + 1), inertia)
        pylab.axis([0, min(self.__data_matrix.shape) + 1, 0, 100])
        pylab.show()
        
    def plot_biplot(self, axes = (0, 1)):
        import pylab
        if len(axes) != 2:
           raise ValueError("Dim tuple must be of length two")
        rows = self.row_factors()[:, axes]
        columns = self.column_factors()[:, axes]
        pylab.plot(numpy.ravel(rows[:, 0]), numpy.ravel(rows[:, 1]), "ro")
        pylab.plot(numpy.ravel(columns[:, 0]), numpy.ravel(columns[:, 1]), "bs")
        
        if self.row_labels:
             for i, coord in enumerate(rows):
                x, y = coord.T
                pylab.text(x, y, self.row_labels[i], horizontalalignment='center')
        if self.col_labels:
             for i, coord in enumerate(columns):
                x, y = coord.T 
                pylab.text(x, y, self.col_labels[i], horizontalalignment='center')                
        pylab.grid()
        pylab.show()
        
        
    """\
    Deprecated interface. Do not use.
    
    """
#    def getPrincipalRowProfilesCoordinates(self, dim = (0, 1)):
#       """ Return principal coordinates of row profiles with respect
#       to principal axis B.
#       
#       :param dim: Defines which principal axes should be taken into account.
#       :type dim: list
#       
#       """
#       deprecation_warning("getPrincipalRowProfilesCoordinates", "row_factors")
#       if len(dim) == 0:
#           raise ValueError("Dim tuple cannot be of length zero")
#       return numpy.array(numpy.take(self.__f, dim, 1))
#   
#    def getPrincipalColProfilesCoordinates(self, dim = (0, 1)):
#       """ Return principal coordinates of column profiles with respect
#       to principal axes A.
#       
#       :param dim: Defines which principal axes should be taken into account.
#       :type dim: list
#        
#       """    
#       deprecation_warning("getPrincipalColProfilesCoordinates", "column_factors")
#       if len(dim) == 0:
#           raise ValueError("dim tuple cannot be of length zero")
#       return numpy.array(numpy.take(self.__g, dim, 1))
#    
#    def getStandardRowCoordinates(self):
#        deprecation_warning("getStandardRowCoordinates", "standard_row_factors")
#        dinv = numpy.where(self.__d != 0, 1. / self.__d, 0)
#        return numpy.matrixmultiply(self.__f, numpy.transpose(dinv))
#    
#    def getStandardColCoordinates(self):
#        deprecation_warning("getStandardColCoordinates", "standard_column_factors")
#        dinv = numpy.where(self.__d != 0, 1. / self.__d, 0)
#        return numpy.matrixmultiply(self.__g, dinv)
#    
#    def DecompositionOfInertia(self, rows=False):
#        """ Returns decomposition of the inertia across the principal axes.
#        Columns of this matrix represent contribution of the rows or columns
#        to the inertia of axes. If ``rows`` is True inertia is
#        decomposed across rows, across columns if False.
#        
#        """
#        if rows:
#            deprecation_warning("DecompositionOfInertia", "row_inertia")
#            return numpy.multiply(self.__row_sums, numpy.multiply(self.__f, self.__f))
#        else:
#            deprecation_warning("DecompositionOfInertia", "column_inertia")
#            return numpy.multiply(numpy.transpose(self.__col_sums), numpy.multiply(self.__g, self.__g))
#        
#    def InertiaOfAxis(self, percentage = False):
#        """ Return :obj:`numpy.ndarray` whose elements are inertias of axes.
#        If ``percentage`` is True percentages of inertias of each axis are returned.
#        
#        """
#        deprecation_warning("InertiaOfAxis", "inertia_of_axes")
#        inertias = numpy.array(numpy.sum(self.DecompositionOfInertia(), 0).tolist()[0])
#        if percentage:
#            return inertias / numpy.sum(inertias) * 100
#        else:
#            return inertias
#        
#    def ContributionOfPointsToAxis(self, rowColumn = 0, axis = 0, percentage = 0):
#        """ Returns :obj:`numpy.ndarray` whose elements are contributions
#        of points to the inertia of axis. Argument ``rowColumn`` defines if
#        the calculation will be performed for row (default action) or
#        column points. The values can be represented in percentages if
#        percentage = 1.
#         
#        """
#        deprecation_warning("ContributionOfPointsToAxis", "nothing")
#        contribution = numpy.array(numpy.transpose(self.DecompositionOfInertia(rowColumn)[:,axis]).tolist()[0])
#        if percentage:
#            return contribution / numpy.sum(contribution) * 100
#        else:
#            return contribution
#        
#    def PointsWithMostInertia(self, rowColumn = 0, axis = (0, 1)):
#        """ Returns indices of row or column points sorted in decreasing
#        value of their contribution to axes defined in a tuple axis.
#        
#        """
#        deprecation_warning("PointsWithMostInertia", "ordered_row_indices")
#        contribution = self.ContributionOfPointsToAxis(rowColumn = rowColumn, axis = axis[0], percentage = 0) + \
#                        self.ContributionOfPointsToAxis(rowColumn = rowColumn, axis = axis[1], percentage = 0)
#        tmp = zip(range(len(contribution)), contribution)
#
#        tmp.sort(lambda x, y: cmp(x[1], y[1]))
#
#        a = [i for (i, v) in tmp]
#        a.reverse()
#        return a
    
    
def burt_table(data, attributes):
    """ Construct a Burt table (all values cross-tabulation) from data for attributes.
    
    Return and ordered list of (attribute, value) pairs and a numpy.ndarray with the tabulations.
    
    :param data: Data table.
    :type data: :class:`Orange.data.Table`
    
    :param attributes: List of attributes (must be Discrete).
    :type attributes: list
    
    Example ::
    
        >>> data = Orange.data.Table("smokers_ct")
        >>> items, counts = burt_table(data, [data.domain["Staff group"], data.domain["Smoking category"]])
        
    """
    values = [(attr, value) for attr in attributes for value in attr.values]
    table = numpy.zeros((len(values), len(values)))
    counts = [len(attr.values) for attr in attributes]
    offsets = [sum(counts[: i]) for i in range(len(attributes))]
    for i in range(len(attributes)):
        for j in range(i + 1):
            attr1 = attributes[i]
            attr2 = attributes[j]
            
            cm = contingency.VarVar(attr1, attr2, data)
            cm = numpy.array([list(row) for row in cm])
            
            range1 = range(offsets[i], offsets[i] + counts[i])
            range2 = range(offsets[j], offsets[j] + counts[j])
            start1, end1 = offsets[i], offsets[i] + counts[i]
            start2, end2 = offsets[j], offsets[j] + counts[j]
            
            table[start1: end1, start2: end2] += cm
            if i != j: #also fill the upper part
                table[start2: end2, start1: end1] += cm.T
                
    return values, table
    
if __name__ == '__main__':
    a = numpy.random.random_integers(0, 100, 100).reshape(10,-1)
    c = CA(a)
    c.plot_scree_diagram()
    c.plot_biplot()

##    data = matrix([[72,    39,    26,    23 ,    4],
##    [95,    58,    66,    84,    41],
##    [80,    73,    83,     4 ,   96],
##    [79,    93,    35,    73,    63]])
##
##    data = [[9, 11, 4], 
##                [ 3,          5,          3], 
##                [     11,          6,          3], 
##                [24,         73,         48]] 

    # Author punctuation (from 'Correspondence Analysis - Herve Abdi Lynne J. Williams')
    data = [[7836,   13112,  6026 ],
            [53655,  102383, 42413],
            [115615, 184541, 59226],
            [161926, 340479, 62754],
            [38177,  105101, 12670],
            [46371,  58367,  14299]]
    
    c = CA(data, ["Rousseau", "Chateaubriand", "Hugo", "Zola", "Proust","Giraudoux"],
                 ["period", "comma", "other"])
    c.plot_scree_diagram()
    c.plot_biplot()

    import Orange
    data = Orange.data.Table("../../doc/datasets/smokers_ct")
    staff = data.domain["Staff group"]
    smoking = data.domain["Smoking category"]
    cont = contingency.VarVar(staff, smoking, data)
    
    c = CA(cont, staff.values, smoking.values)
    c.plot_scree_diagram()
    c.plot_biplot()
    