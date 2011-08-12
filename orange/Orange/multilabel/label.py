import Orange

def get_labels(data,example):
    """ get the list of labels of the example, using 0,1 to indicate whether the example is belong to this label """
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    
    return [example[i] for i, var in enumerate(data.domain.variables) if var.attributes.has_key('label')]

def get_num_labels(data):
    """ get the number of labels in the data """
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    
    return len( [i for i, var in enumerate(data.domain.variables)
          if var.attributes.has_key('label')])

def is_multilabel(data):
    """ Judge whether the data is multi-label data; if so, return 1, else return 0"""
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    if get_num_labels(data) > 0:
        return 1
    return 0

def get_label_indices(data):
    """ get the index array of the label in the attributes in an increasing order. """
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    
    return [i for i, var in enumerate(data.domain.variables)
          if var.attributes.has_key('label')]

def get_label_names(data):
    """ get a list of label names """
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    
    return [var.name for i, var in enumerate(data.domain.variables)
          if var.attributes.has_key('label')]

def remove_indices(data,indices_to_remove):
    """ remove the attributes in the data according to indices_to_remove, (the indices are sorted from small to large order) """
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    
    domain = data.domain
    newdomain = []
    index = 0
    id = 0
    while id < len(domain.variables):
        if index < len(indices_to_remove):
            while id < indices_to_remove[index]:
                newdomain.append(domain[id])
                id = id + 1  
            else:
                index = index + 1
                id = id + 1
        else:
            newdomain.append(domain[id])
            id = id + 1
    return newdomain

def remove_labels(data):
    """ remove the label attributes in the data"""
    domain = data.domain
    newdomain =  [domain[i] for i, var in enumerate(data.domain.variables)
          if not var.attributes.has_key('label')]
    new_data = data.translate(newdomain)
    return new_data

def get_label_bitstream(data,example):
    """ get the labels in a 0/1 string. For example, if the first char in the string is '1', then the example belongs to the first label"""
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    
    value = ''
    for i, var in enumerate(data.domain.variables):
        if var.attributes.has_key('label'):
            value += example[var].value
    return value      