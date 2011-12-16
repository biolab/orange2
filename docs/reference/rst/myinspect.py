from inspect import *
ofas = formatargspec

def delete_while_accurate(s, accuracy=0.0000001):
    """ Shortens a string containing a float
    while still keeping the desired accuracy. """
    orig = float(s)
    cur = s
    try:
        while abs((float(cur[:-1])-orig)/orig) < accuracy:
            cur = cur[:-1]
    except:
        pass
    return cur

def nfas(*args, **kwargs):

    def fv(value):
        if isinstance(value, float):
            cand = str(value)
            value = delete_while_accurate(cand)
        return "=" + str(value)

    if "formatvalue" not in kwargs and len(args) < 8:
        kwargs["formatvalue"] = fv

    return ofas(*args, **kwargs)

formatargspec = nfas

# inspect.getmembers from Python 2.7
# In Python 2.6 there is try/except missing and function fails on C++-based classes if they define attributes
# Example: AttributeError: 'Plot' object attribute 'animate_points' is an instance attribute
def getmembers27(object, predicate=None):
    """Return all members of an object as (name, value) pairs sorted by name.
    Optionally, only return members that satisfy a given predicate."""
    results = []
    for key in dir(object):
        try:
            value = getattr(object, key)
        except AttributeError:
            continue
        if not predicate or predicate(value):
            results.append((key, value))
    results.sort()
    return results

getmembers = getmembers27

if __name__ == "__main__":
    for a in [ "0", "0.", "-1.3", "1.30000000001" ]:
        print a, "shorter:", delete_while_accurate(a)
