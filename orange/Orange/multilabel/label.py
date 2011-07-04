import Orange

def get_labels(data,example):
    """ get the label list in the example, using 0,1 to indicate whether the example is belong to the label """
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
    """ Judge whether the data is multi-label, if so, return 1, else return 0"""
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    if not data.domain.classVar and get_num_labels(data) > 0:
        return 1
    return 0

def get_label_indices(data):
    """ get an array containing the indexes of the label attributes within the object of the training data in increasing order. """
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    
    return [i for i, var in enumerate(data.domain.variables)
          if var.attributes.has_key('label')]

def get_label_names(data):
    """ get the list of all label names """
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    
    return [var.name for i, var in enumerate(data.domain.variables)
          if var.attributes.has_key('label')]

def remove_indices(data,indicesToRemove):
    """ remove the attributes in the data according to indicesToRemove, (the indices are sorted from small to big) """
    if not isinstance(data, Orange.data.Table):
        raise TypeError('data must be of type \'Orange.data.Table\'')
    
    domain = data.domain
    newdomain = []
    index = 0
    id = 0
    while id < len(domain.variables):
        if index < len(indicesToRemove):
            while id < indicesToRemove[index]:
                newdomain.append(domain[id])
                id = id + 1  
            else:
                index = index + 1
                id = id + 1
        else:
            newdomain.append(domain[id])
            id = id + 1
    return newdomain
            