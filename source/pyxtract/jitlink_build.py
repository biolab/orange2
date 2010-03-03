import re, sys

re_funcdef = re.compile(r"extern[^(]*\(\*i__(?P<cname>[^)]+)\)[^;]+;(\s*//\s*AS\s+(?P<dllfname>.*))")
if sys.platform == "win32":
    slext = ".dll"
else:
    slext = ".so"

dllname = hppname = None
for arg in sys.argv:
    if arg[-len(slext):] == slext:
        dllname = arg
    elif arg[-4:] == ".hpp":
        hppname = arg

if not dllname:
    print "shared library name not given"
    sys.exit(1)

if not hppname:
    print "hpp file not given"
    sys.exit(1)

dllnice = dllname[:-len(slext)]

functions = []
for r in file(hppname):
    mo = re_funcdef.search(r)
    if mo:
        cname, dllfname = mo.group("cname", "dllfname")
        if not dllfname:
            dllfname = cname
        functions.append((cname, dllfname, r))

f = file(hppname[:-4]+".ipp", "wt")

for func,dllfunc,line in functions:
    f.write("#define %s (*i__%s)\n" % (func, func))
f.close()

f = file(hppname[:-4]+".jpp", "wt")
f.write('#include "errors.hpp"\n#include "%s"\n#include "jit_linker.hpp"\n\n' % hppname)

for func,dllfunc,line in functions:
    f.write(line.replace("extern ", "")+"\n")

f.write('\nTJitLink %sLinks[] = {\n' % dllnice)
for func,dllfunc,line in functions:
    f.write('  { (void **)&i__%s, "%s"},\n' % (func, dllfunc))
f.write("""{ NULL, NULL}
};

void %(dll)s_unavailable(...) {
  raiseErrorWho("%(dll)s", "library not found or wrong version");
}

int %(dll)s_status = jit_link("%(dll)s", %(dll)sLinks, %(dll)s_unavailable);
""" % {"dll": dllnice})

f.close()
