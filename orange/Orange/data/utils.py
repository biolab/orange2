"""\
**************************
Data Utilities (``utils``)
**************************

Common operations on :class:`Orange.data.Table`.

"""
from __future__ import absolute_import

from collections import defaultdict

from Orange.data import Table, Domain 

def table_map(table, attrs, exclude_special=True):
    map = defaultdict(list)
    for i, ex in enumerate(table):
        key = [ex[a] for a in attrs]
        if exclude_special and any(k.isSpecial() for k in key):
            continue
        key = tuple([str(k) for k in key])
        map[key].append(i)
    return map
    
def join_domains(domain1, domain2):
    variables = domain1.variables + domain1.variables
    used_set = set()
    def used(vars):
        mask = []
        for var in vars:
            mask.append(var not in used_set)
            used_set.add(var)
            
    used_mask1 = used(domain1.variables)
    used_mask2 = used(domain2.variables)
    if domain2.classVar:
        used_mask2[-1] = True
        
    variables = [v for v, used in zip(variables, used_mask1 + used_mask2)]
    
    joined_domain = Domain(variables, domain2.classVar)
    joined_domain.add_metas(domain1.get_metas())
    joined_domain.add_metas(domain2.get_metas())
    return joined_domain, used_mask1, used_mask2
    
def left_join(table1, table2, on_attrs1, on_attrs2):
    """ Left join table1 and table2 on attributes attr1 and attr2
    """
    if not isinstance(on_attrs1, (list, tuple)):
        on_attrs1 = [on_attrs1]
    if not isinstance(on_attrs2, (list, tuple)):
        on_attrs2 = [on_attrs2]
    key_map1 = table_map(table1, on_attrs1)
    key_map2 = table_map(table2, on_attrs2)
    domain1, domain2 = table1.domain, table2.domain
    
    left_examples = []
    right_examples = []
    for ex in table1:
        key = tuple([str(ex[a]) for a in on_attrs1])
        if key in key_map1 and key in key_map2:
            for ind in key_map2[key]:
                ex2 = table2[ind]
                left_examples.append(ex)
                right_examples.append(ex2)
                
    left_table = Table(left_examples)
    right_table = Table(right_examples)
    new_table = Table([left_table, right_table])
    return new_table
    
def right_join(table1, table2, on_attrs1, on_attrs2):
    """ Right join table1 and table2 on attributes attr1 and attr2
    """
    if not isinstance(on_attrs1, (list, tuple)):
        on_attrs1 = [on_attrs1]
    if not isinstance(on_attrs2, (list, tuple)):
        on_attrs2 = [on_attrs2]
    key_map1 = table_map(table1, on_attrs1)
    key_map2 = table_map(table2, on_attrs2)
    domain1, domain2 = table1.domain, table2.domain

    left_examples = []
    right_examples = []
    for ex in table2:
        key = tuple([str(ex[a]) for a in on_attrs2])
        if key in key_map1 and key in key_map2:
            for ind in key_map1[key]:
                ex1 = table1[ind]
                left_examples.append(ex1)
                right_examples.append(ex)
                
    left_table = Table(left_examples)
    right_table = Table(right_examples)
    new_table = Table([left_table, right_table])
    return new_table
    
def hstack(table1, table2):
    """ Horizontally stack ``table1`` and ``table2`` 
    """
    return Table([table1, table2])

def vstack(table1, table2):
    """ Stack ``table1`` and ``table2`` vertically.
    """
    return Table(table1[:] + table2[:])

def take(table, indices, axis=0):
    """ Take values form the ``table`` along the ``axis``. 
    """
    indices = mask_to_indices(indices, (len(table), len(table.domain)), axis)
    if axis == 0:
        # Take the rows (instances)
        instances = [table[i] for i in indices]
        table = Table(instances) if instances else Table(table.domain)
    elif axis == 1:
        # Take the columns (attributes)
        variables = table.domain.variables
        vars = [variables[i] for i in indices]
        domain = Domain(vars, table.domain.class_var in vars)
        domain.add_metas(table.domain.get_metas())
        table = Table(domain, table)
    return table

def mask_to_indices(mask, shape, axis=0):
    """ Convert a mask into indices.
    """
    import numpy
    mask = numpy.asarray(mask)
    dtype = mask.dtype
    size = shape[axis]
    if dtype.kind == "b":
        if len(mask) != size:
            raise ValueError("Mask size does not match the shape.")
        indices = [i for i, m in zip(range(size), mask)]
    elif dtype.kind == "i":
        indices = mask
    return indices


from threading import Lock as _Lock
_global_id = 0
_global_id_lock = _Lock()
 
def range_generator():
    global _global_id
    while True:
        with _global_id_lock:
            id = int(_global_id)
            _global_id += 1
        yield id
        
def uuid_generator():
    import uuid
    while True:
        yield str(uuid.uuid4())

from Orange.data import new_meta_id, variable

_row_meta_id = new_meta_id()
_id_variable = variable.String("Row Id")

def add_row_id(table, start=0):
    """ Add an Row Id meta variable to the table.
    
    Parameters
    ==========
    table : Orange.data.Table
        The ids will be added to this table.
    start : int
        Start id for the ids. It can also be an iterator yielding
        unique values.
        
    """
    if _row_meta_id not in table.domain.get_metas():
        table.domain.add_meta(_row_meta_id, _id_variable)
        
    if isinstance(start, int):
        ids = iter(range(start, start + len(table)))
    else:
        ids = start
                
    for ex in table:
        ex[_id_variable] = str(ids.next())
    
        
    

