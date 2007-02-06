import time, copy, orange
from string import *

class Context:
    def __init__(self, **argkw):
        self.time = time.time()
        self.__dict__.update(argkw)

    def __getstate__(self):
        s = dict(self.__dict__)
        for nc in getattr(self, "noCopy", []):
            if s.has_key(nc):
                del s[nc]
        return s

    
class ContextHandler:
    maxSavedContexts = 50
    
    def __init__(self, contextName = "", cloneIfImperfect = True, loadImperfect = True, findImperfect = True, syncWithGlobal = True, **args):
        self.contextName = contextName
        self.localContextName = "localContexts"+contextName
        self.cloneIfImperfect, self.loadImperfect, self.findImperfect = cloneIfImperfect, loadImperfect, findImperfect
        self.syncWithGlobal = syncWithGlobal
        self.globalContexts = []
        self.__dict__.update(args)

    def newContext(self):
        return Context()

    def openContext(self, widget, *arg, **argkw):
        context, isNew = self.findOrCreateContext(widget, *arg, **argkw)
        if context:
            if isNew:
                self.settingsFromWidget(widget, context)
            else:
                self.settingsToWidget(widget, context)
        return context

    def initLocalContext(self, widget):
        if not hasattr(widget, self.localContextName):
            if self.syncWithGlobal:
                setattr(widget, self.localContextName, self.globalContexts)
            else:
                setattr(widget, self.localContextName, copy.deepcopy(self.globalContexts))
        
    def findOrCreateContext(self, widget, *arg, **argkw):        
        index, context, score = self.findMatch(widget, self.findImperfect and self.loadImperfect, *arg, **argkw)
        if context:
            if index < 0:
                self.addContext(widget, context)
            else:
                self.moveContextUp(widget, index)
            return context, False
        else:
            context = self.newContext()
            self.addContext(widget, context)
            return context, True

    def closeContext(self, widget, context):
        self.settingsFromWidget(widget, context)

    def fastSave(self, context, widget, name):
        pass

    def settingsToWidget(self, widget, context):
        cb = getattr(widget, "settingsToWidgetCallback" + self.contextName, None)
        return cb and cb(self, context)

    def settingsFromWidget(self, widget, context):
        cb = getattr(widget, "settingsFromWidgetCallback" + self.contextName, None)
        return cb and cb(self, context)

    def findMatch(self, widget, imperfect = True, *arg, **argkw):
        bestI, bestContext, bestScore = -1, None, -1
        for i, c in enumerate(getattr(widget, self.localContextName)):
            score = self.match(c, imperfect, *arg, **argkw)
            if score == 2:
                return i, c, score
            if score and score > bestScore:
                bestI, bestContext, bestScore = i, c, score

        if bestContext and self.cloneIfImperfect:
            if hasattr(self, "cloneContext"):
                bestContext = self.cloneContext(bestContext, *arg, **argkw)
            else:
                import copy
                bestContext = copy.deepcopy(bestContext)
            bestI = -1
                
        return bestI, bestContext, bestScore
            
    def moveContextUp(self, widget, index):
        localContexts = getattr(widget, self.localContextName)
        l = getattr(widget, self.localContextName)
        l.insert(0, l.pop(index))

    def addContext(self, widget, context):
        l = getattr(widget, self.localContextName)
        l.insert(0, context)
        while len(l) > self.maxSavedContexts:
            del l[-1]

    def mergeBack(self, widget):
        if not self.syncWithGlobal:
            self.globalContexts.extend(getattr(widget, self.localContextName))
            self.globalContexts.sort(lambda c1,c2: -cmp(c1.time, c2.time))
            self.globalContexts = self.globalContexts[:self.maxSavedContexts]


class ContextField:
    def __init__(self, name, flags = 0, **argkw):
        self.name = name
        self.flags = flags
        self.__dict__.update(argkw)
    

class ControlledAttributesDict(dict):
    def __init__(self, master):
        self.master = master

    def __setitem__(self, key, value):
        if not self.has_key(key):
            dict.__setitem__(self, key, [value])
        else:
            dict.__getitem__(self, key).append(value)
        self.master.setControllers(self.master, key, self.master, "")
        
        
class DomainContextHandler(ContextHandler):
    Optional, SelectedRequired, Required = range(3)
    RequirementMask = 3
    NotAttribute = 4
    List = 8
    RequiredList = Required + List
    SelectedRequiredList = SelectedRequired + List

    MatchValuesNo, MatchValuesClass, MatchValuesAttributes = range(3)
    
    def __init__(self, contextName, fields = [],
                 cloneIfImperfect = True, loadImperfect = True, findImperfect = True, syncWithGlobal = True, maxAttributesToPickle = 100, matchValues = 0, **args):
        ContextHandler.__init__(self, contextName, cloneIfImperfect, loadImperfect, findImperfect, syncWithGlobal, **args)
        self.maxAttributesToPickle = maxAttributesToPickle
        self.matchValues = matchValues
        self.fields = []
        for field in fields:
            if isinstance(field, ContextField):
                self.fields.append(field)
            elif type(field)==str:
                self.fields.append(ContextField(field, self.Required))
            # else it's a tuple
            else:
                flags = field[1]
                if type(field[0]) == list:
                    self.fields.extend([ContextField(x, flags) for x in field[0]])
                else:
                    self.fields.append(ContextField(field[0], flags))
        
    def encodeDomain(self, domain):
        if self.matchValues == 2:
            d = dict([(attr.name, attr.varType != orange.VarTypes.Discrete and attr.varType or attr.values)
                         for attr in domain])
            d.update(dict([(attr.name, attr.varType != orange.VarTypes.Discrete and attr.varType or attr.values)
                         for attr in domain.getmetas().values()]))
        else:
            d = dict([(attr.name, attr.varType) for attr in domain.attributes])
            classVar = domain.classVar
            if classVar:
                if self.matchValues and classVar.varType == orange.VarTypes.Discrete:
                    d[classVar.name] = classVar.values
                else:
                    d[classVar.name] = classVar.varType

            d.update(dict([(attr.name, attr.varType) for attr in domain.getmetas().values()]))

        return d
    
    def findOrCreateContext(self, widget, domain):
        if not domain:
            return None, False
        if type(domain) != orange.Domain:
            domain = domain.domain
            
        encodedDomain = self.encodeDomain(domain)
        context, isNew = ContextHandler.findOrCreateContext(self, widget, domain, encodedDomain)
        if not context:
            return None, False
        
        context.encodedDomain = encodedDomain

        metaIds = domain.getmetas().keys()
        metaIds.sort()
        context.orderedDomain = [(attr.name, attr.varType) for attr in domain] + [(domain[i].name, domain[i].varType) for i in metaIds]

        if isNew:
            context.values = {}
            context.noCopy = ["orderedDomain"]
        return context, isNew

    def settingsToWidget(self, widget, context):
        ContextHandler.settingsToWidget(self, widget, context)
        excluded = {}
        for field in self.fields:
            name, flags = field.name, field.flags

            excludes = getattr(field, "reservoir", [])
            if excludes:
                if type(excludes) != list:
                    excludes = [excludes]
                for exclude in excludes:
                    excluded.setdefault(exclude, [])

            if not context.values.has_key(name):
                continue
            
            value = context.values[name]

            if not flags & self.List:
                setattr(widget, name, value[0])
                for exclude in excludes:
                    excluded[exclude].append(value)

            else:
                newLabels, newSelected = [], []
                oldSelected = hasattr(field, "selected") and context.values.get(field.selected, []) or []
                for i, saved in enumerate(value):
                    if saved in context.orderedDomain:
                        if i in oldSelected:
                            newSelected.append(len(newLabels))
                        newLabels.append(saved)

                context.values[name] = newLabels
                setattr(widget, name, value)

                if hasattr(field, "selected"):
                    context.values[field.selected] = newSelected
                    setattr(widget, field.selected, context.values[field.selected])

                for exclude in excludes:
                    excluded[exclude].extend(value)

        for name, values in excluded.items():
            ll = filter(lambda a: a not in values, context.orderedDomain)
            setattr(widget, name, ll)
            

    def settingsFromWidget(self, widget, context):
        ContextHandler.settingsFromWidget(self, widget, context)
        context.values = {}
        for field in self.fields:
            if not field.flags & self.List:
                self.saveLow(context, widget, field.name, widget.getdeepattr(field.name))
            else:
                context.values[field.name] = widget.getdeepattr(field.name)
                if hasattr(field, "selected"):
                    context.values[field.selected] = list(widget.getdeepattr(field.selected))

    def fastSave(self, context, widget, name, value):
        if context:
            for field in self.fields:
                if name == field.name:
                    if field.flags & self.List:
                        context.values[field.name] = value
                    else:
                        self.saveLow(context, widget, name, value)
                    return
                if name == getattr(field, "selected", None):
                    context.values[field.selected] = list(value)
                    return

    def saveLow(self, context, widget, field, value):
        if type(value) == str:
            context.values[field] = value, context.encodedDomain.get(value, -1) # -1 means it's not an attribute
        else:
            context.values[field] = value, -2

    def match(self, context, imperfect, domain, encodedDomain):
        if encodedDomain == context.encodedDomain:
            return 2
        if not imperfect:
            return 0

        filled = potentiallyFilled = 0
        for field in self.fields:
            value = context.values.get(field.name, None)
            if value:
                if field.flags & self.List:
                    potentiallyFilled += len(value)
                    if field.flags & self.RequirementMask == self.Required:
                        filled += len(value)
                        for item in value:
                            if encodedDomain.get(item[0], None) != item[1]:
                                return 0
                    else:
                        selectedRequired = field.flags & self.RequirementMask == self.SelectedRequired
                        for i in context.values.get(field.selected, []):
                            if value[i] in encodedDomain:
                                filled += 1
                            else:
                                if selectedRequired:
                                    return 0
                else:
                    potentiallyFilled += 1
                    if value[1] >= 0:
                        if encodedDomain.get(value[0], None) == value[1]:
                            filled += 1
                        else:
                            if field.flags & self.Required:
                                return 0

            if not potentiallyFilled:
                return 1.0
            else:
                return filled / float(potentiallyFilled)

    def cloneContext(self, context, domain, encodedDomain):
        import copy
        context = copy.deepcopy(context)
        
        for field in self.fields:
            value = context.values.get(field.name, None)
            if value:
                if field.flags & self.List:
                    i = j = realI = 0
                    selected = context.values.get(field.selected, [])
                    selected.sort()
                    nextSel = selected and selected[0] or None
                    while i < len(value):
                        if encodedDomain.get(value[i][0], None) != value[i][1]:
                            del value[i]
                            if nextSel == realI:
                                del selected[j]
                                nextSel = j < len(selected) and selected[j] or None
                        else:
                            if nextSel == realI:
                                selected[j] -= realI - i
                                j += 1
                                nextSel = j < len(selected) and selected[j] or None
                            i += 1
                        realI += 1
                    if hasattr(field, "selected"):
                        context.values[field.selected] = selected[:j]
                else:
                    if value[1] >= 0 and encodedDomain.get(value[0], None) != value[1]:
                        del context.values[field.name]
                        
        context.encodedDomain = encodedDomain
        context.orderedDomain = [(attr.name, attr.varType) for attr in domain]
        return context

    # this is overloaded to get rid of the huge domains
    def mergeBack(self, widget):
        if not self.syncWithGlobal:
            self.globalContexts.extend(getattr(widget, self.localContextName))
        mp = self.maxAttributesToPickle
        self.globalContexts = filter(lambda c: len(c.encodedDomain) < mp, self.globalContexts)
        self.globalContexts.sort(lambda c1,c2: -cmp(c1.time, c2.time))
        self.globalContexts = self.globalContexts[:self.maxSavedContexts]

    
class ClassValuesContextHandler(ContextHandler):
    def __init__(self, contextName, fields = [], syncWithGlobal = True, **args):
        ContextHandler.__init__(self, contextName, False, False, False, syncWithGlobal, **args)
        if isinstance(fields, list):
            self.fields = fields
        else:
            self.fields = [fields]
        
    def findOrCreateContext(self, widget, classes):
        if isinstance(classes, orange.Variable):
            classes = classes.varType == orange.VarTypes.Discrete and classes.values
        if not classes:
            return None, False
        context, isNew = ContextHandler.findOrCreateContext(self, widget, classes)
        if not context:
            return None, False
        context.classes = classes
        if isNew:
            context.values = {}
        return context, isNew

    def settingsToWidget(self, widget, context):
        ContextHandler.settingsToWidget(self, widget, context)
        for field in self.fields:
            setattr(widget, field, context.values[field])
            
    def settingsFromWidget(self, widget, context):
        ContextHandler.settingsFromWidget(self, widget, context)
        context.values = dict([(field, widget.getdeepattr(field)) for field in self.fields])

    def fastSave(self, context, widget, name, value):
        if context and name in self.fields:
            context.values[name] = value

    def match(self, context, imperfect, classes):
        return context.classes == classes and 2

    def cloneContext(self, context, domain, encodedDomain):
        import copy
        return copy.deepcopy(context)
        


### Requires the same the same attributes in the same order
### The class overloads domain encoding and matching.
### Due to different encoding, it also needs to overload saveLow and cloneContext
### (the latter gets really simple now).
### We could simplify some other methods, but prefer not to replicate the code
class PerfectDomainContextHandler(DomainContextHandler):
    def __init__(self, contextName, fields = [],
                 syncWithGlobal = True, maxAttributesToPickle = 100, matchValues = 0, **args):
        DomainContextHandler.__init__(self, contextName, fields, False, False, False, syncWithGlobal, **args)

        
    def encodeDomain(self, domain):
        if self.matchValues == 2:
            attributes = tuple([(attr.name, attr.varType != orange.VarTypes.Discrete and attr.varType or attr.values)
                         for attr in domain])
            classVar = domain.classVar
            if classVar:
                classVar = classVar.name, classVar.varType != orange.VarType.Discrete and classVar.varType or classVar.values
            metas = dict([(attr.name, attr.varType != orange.VarTypes.Discrete and attr.varType or attr.values)
                         for attr in domain.getmetas().values()])
        else:
            attributes = tuple([(attr.name, attr.varType) for attr in domain.attributes])
            classVar = domain.classVar
            if classVar:
                classVar = classVar.name, classVar.varType
            metas = dict([(attr.name, attr.varType) for attr in domain.getmetas().values()])
        return attributes, classVar, metas
    


    def match(self, context, imperfect, domain, encodedDomain):
        return encodedDomain == context.encodedDomain and 2


    def saveLow(self, context, widget, field, value):
        if type(value) == str:
            atts = [x for x in context.encodedDomain if x[0] == value]
            context.values[field] = value, (atts and atts[1] or -1) # -1 means it's not an attribute
        else:
            context.values[field] = value, -2


    def cloneContext(self, context, domain, encodedDomain):
        import copy
        context = copy.deepcopy(context)
        
