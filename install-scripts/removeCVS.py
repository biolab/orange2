import sys, os

def removeCVS(dir):
    if os.path.exists(dir+"\\CVS"):
        for fle in os.listdir(dir+"\\CVS"):
            os.remove(dir+"\\CVS\\"+fle)
        os.rmdir(dir+"\\CVS")

    for fle in os.listdir(dir):
        if os.path.isdir(dir+"\\"+fle):
            removeCVS(dir+"\\"+fle)

removeCVS(sys.argv[1])