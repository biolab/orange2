import xml.dom.minidom, re

def _zip_open(zipfile, file, mode='r'):
    if hasattr(zipfile, "open"):
        return zipfile.open(file, mode)
    else:
        from cStringIO import StringIO
        return StringIO(zipfile.read(file))

def create_text_element(tag_name, value):
    result = xml.dom.minidom.Element(tag_name)
    textNode = xml.dom.minidom.Text()
    textNode.data = value
    result.appendChild(textNode)
    return result

def xml_set(parent, node_name, value):
    child = get_element_nonrecursive(parent, node_name)
    if not child:
        if value:
            parent.appendChild(create_text_element(node_name, value))
    else:
        if not value:
            parent.removeChild(child)
        else:
            for text in child.childNodes:
                child.removeChild(text)
            textNode = xml.dom.minidom.Text()
            textNode.data = value
            child.appendChild(textNode)

def xml_text_of(node, parent=None, multiline=False):
    if node.__class__ is str:
        node = get_element_nonrecursive(parent, node)
    t = ""
    if node==None:
        return t
    for n in node.childNodes:
        if n.nodeType == n.COMMENT_NODE:
            continue
        if n.nodeType != n.TEXT_NODE:
            break
        t += n.data
    if multiline:
        t = "\n\n".join( map(lambda x: x.strip(), re.split("\n[ \t]*\n", t.strip())) )
    else:
        t = re.sub("\s+", " ", t.strip())
    return t

def get_element_nonrecursive(parent, elementname, create=False):
    for node in [n for n in parent.childNodes if n.nodeType==n.ELEMENT_NODE]:
        if node.tagName == elementname:
            return node
    if create:
        node = xml.dom.minidom.Element(elementname)
        parent.appendChild(node)
        return node
    return None
    
