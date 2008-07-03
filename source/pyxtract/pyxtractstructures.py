
class ClassDefinition:
  def __init__(self, **argkw):
    self.constructor=self.call=self.basetype=self.description=self.infile=self.datastructure=self.dictfield=None
    self.properties={}
    self.methods={}
    self.specialmethods={}
    self.constants={}
    self.subconstants = {}
    self.hidden = 0
    self.imported = 0
    self.used = 0
    self.abstract = False
    self.allows_empty_args = False
    self.constructor_keywords = []
    self.recognized_attributes = []
    self.__dict__.update(argkw)

  def hasSpecialProperties(self):
    for property in self.properties.values():
      if not property.builtin:
        return 1
    return 0


class ConstructorDefinition:
  def __init__(self, **argkw):
    self.type=None
    self.definition=self.arguments=self.description=None
    self.__dict__.update(argkw)

class CallDefinition:
  def __init__(self, **argkw):
    self.arguments=self.description=None
    self.__dict__.update(argkw)

class AttributeDefinition:
  def __init__(self, **argkw):
    self.description=None
    self.builtin=0
    self.hasget=0
    self.hasset=0
    self.__dict__.update(argkw)

class MethodDefinition:
  def __init__(self, **argkw):
    self.description=self.arguments=self.argkw=None
    self.__dict__.update(argkw)

class FunctionDefinition:
  def __init__(self, **argkw):
    self.cname=self.argkw=self.definition=self.description=self.arguments=None
    self.__dict__.update(argkw)
  
class ConstantDefinition:
  def __init__(self, **argkw):
    self.ccode=self.cfunc=self.description=None
    self.__dict__.update(argkw)
    

def addClassDef(cds, typename, parsedFile, str="", val=1, warn=1):
  """
  If class is not been encountered yet, it creates a new class definition.
  It sets the attribute str to value val.
  If the attribute was non-null and warn=1, it gives a warning.
  """
  
  if not cds.has_key(typename):
    if str:
      cds[typename]=apply(ClassDefinition, (), {str:val, 'infile':parsedFile})
    else:
      cds[typename]=ClassDefinition(infile=parsedFile)
  elif str:
    #if warn and hasattr(cds[typename], str):
    #  print ("Warning: overriding definition of %s for %s", (str, typename))
    setattr(cds[typename], str, val)
    if parsedFile and cds[typename].infile!=parsedFile:
      print ("Warning: %s appears in different files (%s, %s)" % (typename, cds[typename].infile, parsedFile))
