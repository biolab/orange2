.. py:currentmodule:: Orange.data

================================
Symmetric matrix (``SymMatrix``)
================================

Class `Orange.data.SymMatrix` is symmetric (square) matrix of size fixed at 
construction time (and stored to the attribute dim). The constructor expects a 
sequence of sequences (eg. a list of lists, a list of tuples...) or the matrix 
dimension. An optional additional argument gives the default value; the default 
is 0.


For instance, a nice list to initialize the matrix with looks like this:

Example from :obj:`Orange.clustering.hierarchical` (Example 1 - Toy matrix)
::

    import Orange
    from Orange.clustering import hierarchical
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

This matrix, meant as a distance matrix, is used in example from the description 
of hierarchical clustering. SymMatrix also stores the diagonal elements; here 
they are not specified, so they are set to 0. The matrix needn't by so nice. If 
any line was shorter, the missing elements would be set to 0 as well. Any line 
could also be longer, spreading across the diagonal, in which case the 
constructor would check for asymmetries. For instance, if the matrix started 
by
::

	m = [[],
	     [ 3,  0, f],
	     [ 2,  4], ...
     	
this would only be OK if f equals 2; otherwise, the matrix would be asymmetric.

Finally, no line can be longer than the total number of lines. Here we have 10 
rows, so no row may have more than 10 columns.

So much for construction. Indexing is implemented so that order of indices is 
unimportant (unless set otherwise, see below), eg, if m is an instance of 
SymMatrix, then m[2, 4] addresses the same element as m[4, 2].


part of `symmatrix.py`_
    
.. literalinclude:: code/symmatrix.py
    :lines: 1-6

Although we set only the lower left half of the matrix (if we interpret the 
first index, i, as the row index), we have actually constructed a whole 
symmetric matrix.

>>> print m
(( 1.000,  2.000,  3.000,  4.000),
 ( 2.000,  4.000,  6.000,  8.000),
 ( 3.000,  6.000,  9.000, 12.000),
 ( 4.000,  8.000, 12.000, 16.000))
 
Other manipulations also respect the symmetricity, for instance, increasing an 
element m[3, 2] += 15 will also increase m[2, 3] (since this is, in fact, one 
and the same element).

The matrix has an attribute matrixType whose value can be set to SymMatrix.Lower 
(0), SymMatrix.Upper (1), SymMatrix.Symmetric (2, default), 
SymMatrix.Lower_Filled (3), SymMatrix.Upper_Filled (4). By setting it to Lower 
or Upper, we limit the matrix to the lower or upper half; attempts to index 
anything above or below the diagonal, respectively, will yield an error. With 
Lower_Filled and Upper_Field, the elements of the other half (upper or lower, 
respectively) still exist, but are set to zero and can be read, but cannot be 
modified. The matrix type is by default initially set to symmetric, but can be 
changed at any time. If it is, for instance, changed from lower to upper, the 
matrix gets transposed (actually, nothing really happens, the change only 
affects indexing (and printing) while the internal matrix representation stays 
the same, so changing the matrix type takes no time).

If we, for instance, change the matrix type of the above matrix to 
SymMatrix.Upper, it gets printed as

>>> m.matrixType = m.Upper
>>> print m
(( 1.000,  2.000,  3.000,  4.000),
 (         4.000,  6.000,  8.000),
 (                 9.000, 12.000),
 (                        16.000))

Changing the type to SymMatrix.Lower_Filled will change the printout to

>>> m.matrixType = m.Lower_Filled
>>> print m
(( 1.000,  0.000,  0.000,  0.000),
 ( 2.000,  4.000,  0.000,  0.000),
 ( 3.000,  6.000,  9.000,  0.000),
 ( 4.000,  8.000, 12.000, 16.000))

It is also possible to index entire rows by using a single index instead of two.

>>> print m[1]
(3.0, 6.0, 9.0, 0.0)

In the similar manner, you can iterate over the matrix using a for loop.

>>> m.matrixType = m.Lower
>>> for row in m:
...     print row
(1.0,)
(2.0, 4.0)
(3.0, 6.0, 9.0)
(4.0, 8.0, 12.0, 16.0)

Slicing also works, but what you get by taking, for instance, m[:3] is a tuple 
containing the first three lines of the matrix (again represented as tuples).

Started to wonder why always those annoying tuples and not lists that you can 
change as you will? To give you a clear message about one thing you cannot do 
with the matrix: you cannot change its contents by manipulating the rows you get 
by row indexing or slicing. Also, you cannot assign whole rows to matrices:

>>> m[1] = (0, 0, 0, 0)
Traceback (most recent call last):
  File "<interactive input>", line 1, in ?
IndexError: two integer indices expected

If you want to manipulate the row contents for your purposes, knowing that it 
doesn't change the matrix, convert it to list by calling list(m[i]).

.. _symmatrix.py: code/symmatrix.py