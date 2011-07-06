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

if __name__ == "__main__":
    for a in [ "0", "0.", "-1.3", "1.30000000001" ]:
        print a, "shorter:", delete_while_accurate(a)
