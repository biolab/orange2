import os
import doctest
try:
    import unittest2 as unittest
except:
    import unittest
import sys


def ispackage(path):
    return os.path.isdir(path) and \
        os.path.isfile(os.path.join(path, '__init__.py'))


def getpackage(fname):
    if not fname.endswith('.py') and not ispackage(fname):
        return None
    base, ext = os.path.splitext(os.path.basename(fname))
    if base == '__init__':
        mod_parts = []
    else:
        mod_parts = [base]
    path, part = os.path.split(os.path.split(fname)[0])
    while part:
        if ispackage(os.path.join(path, part)):
            mod_parts.append(part)
        else:
            break
        path, part = os.path.split(path)
    mod_parts.reverse()
    return os.path.join(path, part), '.'.join(mod_parts)


def get_files(base_dirs, exts):
    for base_dir in base_dirs:
        for dirpath, _, fnames in os.walk(base_dir):
            for fname in fnames:
                if os.path.splitext(fname)[1] in exts:
                    yield os.path.join(dirpath, fname)


def load_tests(loader, tests, ignore):
    exclude_dirs = ['/orng/', '/doc/', '/unit/', '/testing/', '/multilabel/', '/OrangeWidgets/']
    for fname in get_files(base_dirs, ['.py']):
        if any(dir in fname for dir in exclude_dirs):
            continue
        #print(fname)
        path, package = getpackage(fname)
        sys.path.insert(0, os.path.split(fname)[0])
        __import__(package)
        mod = sys.modules[package]
        try:
            tests.addTests(doctest.DocTestSuite(mod))
        except ValueError:  # Has no tests
            pass
        sys.path.pop(0)
    return tests


if __name__ == '__main__':
    base_dirs = [os.path.abspath('../../classification')]
    unittest.main()
