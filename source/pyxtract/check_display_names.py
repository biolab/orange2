import orange, Orange

for cls in orange.__dict__.values():
    if type(cls) == type:
        try:
            cls2 = eval(cls.__module__+"."+cls.__name__)
        except AttributeError as err:
            print "%s: %s" % (cls.__module__+"."+cls.__name__, err)
            continue
        if cls2 != cls:
            print cls.__module__+"."+cls.__name__
