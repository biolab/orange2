.. py:currentmodule:: Orange.data

================================
Symmetric matrix (``SymMatrix``)
================================

Class `Orange.data.SymMatrix` is symmetric (square) matrix of size fixed at 
construction time (and stored to the attribute dim).

.. class:: SymMatrix

    .. attribute:: dim
	
        Matrix dimension.
            
    .. attribute:: matrix_type 

        Value can be set to SymMatrix.Lower (0), SymMatrix.Upper (1), 
        SymMatrix.Symmetric (2, default), SymMatrix.Lower_Filled (3) or
        SymMatrix.Upper_Filled (4). 
        
        By setting it to Lower or Upper, we limit the matrix to the lower or 
        upper half. Attempts to index anything above or below the diagonal, 
        respectively, will yield an error. With Lower_Filled and Upper_Field, 
        the elements of the other half (upper or lower, respectively) still 
        exist, but are set to zero and can be read, but cannot be modified. The 
        matrix type is by default initially set to symmetric, but can be changed 
        at any time.

        If matrix type is changed to SymMatrix.Upper, it gets printed as

        >>> m.matrix_type = m.Upper
        >>> print m
        (( 1.000,  2.000,  3.000,  4.000),
         (         4.000,  6.000,  8.000),
         (                 9.000, 12.000),
         (                        16.000))

        Changing the type to SymMatrix.Lower_Filled will change the printout to

        >>> m.matrix_type = m.Lower_Filled
        >>> print m
        (( 1.000,  0.000,  0.000,  0.000),
         ( 2.000,  4.000,  0.000,  0.000),
         ( 3.000,  6.000,  9.000,  0.000),
         ( 4.000,  8.000, 12.000, 16.000))
	
    .. method:: __init__(dim[, default_value])

        Construct an empty symmetric matrix with the given dimension.

        :param dim: matrix dimension
        :type dim: int

        :param default_value: default value (0 by default)
        :type default_value: double
        
        
    .. method:: __init__(instances)

        Construct a new symmetric matrix containing the given data instances. 
        These can be given as Python list containing lists or tuples.

        :param instances: data instances
        :type instances: list of lists
        
        The following example fills the symmetric matrix created above with
        some data from a list::

            import Orange
            m = [[],
                 [ 3],
                 [ 2, 4],
                 [17, 5, 4],
                 [ 2, 8, 3, 8],
                 [ 7, 5, 10, 11, 2],
                 [ 8, 4, 1, 5, 11, 13],
                 [ 4, 7, 12, 8, 10, 1, 5],
                 [13, 9, 14, 15, 7, 8, 4, 6],
                 [12, 10, 11, 15, 2, 5, 7, 3, 1]]
                    
            matrix = Orange.data.SymMatrix(m)

        SymMatrix also stores the diagonal elements. Here they are not 
        specified, so they are set to 0. If any line was shorter, the missing 
        elements would be set to 0 as well. Any line could also be longer, 
        spreading across the diagonal, in which case the constructor would check
        for asymmetries. For instance, if the matrix started by::

            m = [[],
                 [ 3,  0, f],
                 [ 2,  4], ...
    
        this would only be OK if f equals 2; otherwise, the matrix would be 
        asymmetric.

        Finally, no line can be longer than the total number of lines. Here we 
        have 10 rows, so no row may have more than 10 columns.

    .. method:: get_values()
    
        Return all matrix values in a Python list.

    .. method:: get_KNN(i, k)
    
        Return k columns with the lowest value in the i-th row. 
        
        :param i: i-th row
        :type i: int
        
        :param k: number of neighbors
        :type k: int
        
    .. method:: avg_linkage(clusters)
    
        Return a symmetric matrix with average distances between given clusters.  
      
        :param clusters: list of clusters
        :type clusters: list of lists
        
    .. method:: invert(type)
    
        Invert values in the symmetric matrix.
        
        :param type: 0 (-X), 1 (1 - X), 2 (max - X), 3 (1 / X)
        :type type: int

    .. method:: normalize(type)
    
        Normalize values in the symmetric matrix.
        
        :param type: 0 (normalize to [0, 1] interval), 1 (Sigmoid)
        :type type: int
        
        
-------------------
Indexing
-------------------

Indexing is implemented so that order of indices is unimportant (unless set 
otherwise with the matrix_type attribute). For example, if m is an instance of 
SymMatrix, then m[2, 4] addresses the same element as m[4, 2].

`symmatrix.py`_
    
.. literalinclude:: code/symmatrix.py
    :lines: 1-6

Although only the lower left half of the matrix is set , we have actually 
constructed a whole symmetric matrix.

>>> print m
(( 1.000,  2.000,  3.000,  4.000),
 ( 2.000,  4.000,  6.000,  8.000),
 ( 3.000,  6.000,  9.000, 12.000),
 ( 4.000,  8.000, 12.000, 16.000))
 
Other manipulations also respect the symmetricity, for instance, increasing an 
element m[3, 2] += 15 will also increase m[2, 3] (since this is, in fact, one 
and the same element).

Index entire rows by using a single index instead of two:

>>> print m[1]
(3.0, 6.0, 9.0, 0.0)

Iterate over the matrix using a for loop:

>>> m.matrix_type = m.Lower
>>> for row in m:
...     print row
(1.0,)
(2.0, 4.0)
(3.0, 6.0, 9.0)
(4.0, 8.0, 12.0, 16.0)

Slicing also works. For example, m[:3] is a tuple containing the first three 
lines of the matrix (again represented as tuples).

.. _symmatrix.py: code/symmatrix.py