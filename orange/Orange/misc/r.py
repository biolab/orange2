"""
=======================
R compatibility (``r``)
=======================

.. index:: R

Conversion of Orange's structure into R objects (with rpy2 package).

.. autofunction:: dataframe

.. autofunction:: variable_to_vector

.. autofunction:: matrix
"""

import Orange

import rpy2.robjects as robjects
import rpy2.rlike.container as rlc

if hasattr(robjects, "DataFrame"): # rpy2 version 2.1
    DataFrame = robjects.DataFrame
else: # rpy2 version 2.0
    DataFrame = robjects.RDataFrame
    
if hasattr(robjects, "Matrix"): # rpy2 version 2.1
    Matrix = robjects.Matrix
else: # rpy2 version 2.0
    Matrix = robjects.RMatrix
    
if hasattr(robjects, "globalenv"): # rpy2 version 2.1
    globalenv = robjects.globalenv
else: # rpy2 version 2.0
    globalenv = robjects.globalEnv
    
NA_Real = robject.NA_Real if hasattr(robjects, "NA_Real") \
    else robjects.r("NA") #rpy 2.0
NA_Int = robject.NA_Integer if hasattr(robjects, "NA_Integer") \
    else robjects.r("NA") #rpy 2.0
NA_Char = robject.NA_Integer if hasattr(robjects, "NA_Character") \
    else robjects.r("NA") #rpy 2.0

def variable_to_vector(data, attr):
    """
    Convert a Variable to R's vector.
    """
    def cv(value, fn, na):
        return na if value.isSpecial() else fn(value)
    
    if attr.var_type == Orange.data.variable.Variable.Continuous:
        return robjects.FloatVector([cv(ex[attr], float, NA_Real) for ex in data])
    elif attr.var_type == Orange.data.variable.Variable.Discrete:
        return robjects.r("factor")(robjects.StrVector([cv(ex[attr], str, NA_Char) for ex in data]))
    elif attr.var_type == Orange.data.variable.Variable.String:
        return robjects.StrVector([cv(ex[attr], str, NA_Char) for ex in data])
    return vec

def dataframe(data):
    """
    Convert an Orange.data.Table to R's DataFrame.
    """
    attrs = [ attr for attr in data.domain.variables if attr.var_type in \
             [ Orange.data.variable.Variable.Continuous, 
               Orange.data.variable.Variable.Discrete, 
               Orange.data.variable.Variable.String ] ]
       
    odata = []
    for attr in attrs:
        odata.append((attr.name, variable_to_vector(data, attr)))
       
    r_obj = DataFrame(rlc.TaggedList([v for _,v in odata], [t for t,_ in odata]))
    return r_obj

def matrix(matrix):
    """
    Convert an SymMatrix to R's Matrix.
    """
    v = robjects.FloatVector([e for row in matrix for e in row])

    r_obj = robjects.r['matrix'](v, nrow=matrix.dim)
    return r_obj

if __name__ == "__main__":
    data = Orange.data.Table("titanic")
    df = dataframe(data)
    print df

