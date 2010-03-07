import xml.dom.minidom, re

def _zipOpen(zipFile, file, mode='r'):
    if hasattr(zipFile, "open"):
        return zipFile.open(file, mode)
    else:
        from cStringIO import StringIO
        return StringIO(zipFile.read(file))

def createTextElement(tagName, value):
    result = xml.dom.minidom.Element(tagName)
    textNode = xml.dom.minidom.Text()
    textNode.data = value
    result.appendChild(textNode)
    return result

def xmlSet(parent, nodeName, value):
    child = getElementNonRecursive(parent, nodeName)
    if not child:
        if value:
            parent.appendChild(createTextElement(nodeName, value))
    else:
        if not value:
            parent.removeChild(child)
        else:
            for text in child.childNodes:
                child.removeChild(text)
            textNode = xml.dom.minidom.Text()
            textNode.data = value
            child.appendChild(textNode)
        
    
        

def xmlTextOf(node, parent=None, multiLine=False):
    if node.__class__ is str:
        node = getElementNonRecursive(parent, node)
    t = ""
    if node==None:
        return t
    for n in node.childNodes:
        if n.nodeType == n.COMMENT_NODE:
            continue
        if n.nodeType != n.TEXT_NODE:
            break
        t += n.data
    if multiLine:
        t = "\n\n".join( map(lambda x: x.strip(), re.split("\n[ \t]*\n", t.strip())) )
    else:
        t = re.sub("\s+", " ", t.strip())
    return t

def getElementNonRecursive(parent, elementName, create=False):
    for node in [n for n in parent.childNodes if n.nodeType==n.ELEMENT_NODE]:
        if node.tagName == elementName:
            return node
    if create:
        node = xml.dom.minidom.Element(elementName)
        parent.appendChild(node)
        return node
    return None
    
