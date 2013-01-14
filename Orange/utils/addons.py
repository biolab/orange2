from __future__ import absolute_import
"""
==============================
Add-on Management (``addons``)
==============================

.. index:: add-ons

Orange.utils.addons module provides a framework for Orange add-on management. As
soon as it is imported, the following initialization takes place: the list of
installed add-ons is loaded, their directories are added to python path
(:obj:`sys.path`) the callback list is initialized, the stored repository list is
loaded. The most important consequence of importing the module is thus the
injection of add-ons into the namespace.

"""

#TODO Document this module.

import socket
import shelve
import xmlrpclib
import warnings
import re
import pkg_resources
import tempfile
import tarfile
import shutil
import os
import sys
import platform
from collections import namedtuple, defaultdict
from contextlib import closing

import Orange.utils.environ

ADDONS_ENTRY_POINT="orange.addons"

socket.setdefaulttimeout(120)  # In seconds.

OrangeAddOn = namedtuple('OrangeAddOn', ['name', 'available_version', 'installed_version', 'summary', 'description',
                                         'author', 'docs_url', 'keywords', 'homepage', 'package_url',
                                         'release_url', 'release_size', 'python_version'])
#It'd be great if we could somehow read a list and descriptions of widgets, show them in the dialog and enable
#search of add-ons based on keywords in widget names and descriptions.

INDEX_RE = "[^a-z0-9-']"  # RE for splitting entries in the search index

AOLIST_FILE = os.path.join(Orange.utils.environ.orange_settings_dir, "addons.shelve")
def open_addons():
    try:
        addons = shelve.open(AOLIST_FILE, 'c')
        if any(name != name.lower() for name, record in addons.items()):  # Try to read the whole list and check for sanity.
            raise Exception("Corrupted add-on list.")
    except:
        if os.path.isfile(AOLIST_FILE):
            os.remove(AOLIST_FILE)
        addons = shelve.open(AOLIST_FILE, 'n')
    return addons

global addons_corrupted
with closing(open_addons()) as addons:
    addons_corrupted = len(addons)==0

addon_refresh_callback = []

global index
index = defaultdict(list)
def rebuild_index():
    global index

    index = defaultdict(list)
    with closing(open_addons()) as addons:
        for name, ao in addons.items():
            for s in [name, ao.summary, ao.description, ao.author] + (ao.keywords if ao.keywords else []):
                if not s:
                    continue
                words = [word for word in re.split(INDEX_RE, s.lower())
                         if len(word)>1]
                for word in words:
                    for i in range(len(word)):
                        index[word[:i+1]].append(name)

def search_index(query):
    global index
    result = set()
    words = [word for word in re.split(INDEX_RE, query.lower()) if len(word)>1]
    if not words:
        with closing(open_addons()) as addons:
            return addons.keys()
    for word in words:
        result.update(index[word])
    return result

def refresh_available_addons(force=False, progress_callback=None):
    pypi = xmlrpclib.ServerProxy('http://pypi.python.org/pypi')
    if progress_callback:
        progress_callback(1, 0)

    pkg_dict = {}
    for data in pypi.search({'keywords': 'orange'}):
        name = data['name']
        order = data['_pypi_ordering']
        if name not in pkg_dict or pkg_dict[name][0] < order:
            pkg_dict[name] = (order, data['version'])

    try:
        import slumber
        readthedocs = slumber.API(base_url='http://readthedocs.org/api/v1/')
    except:
        readthedocs = None

    global addons_corrupted
    docs = {}
    if progress_callback:
        progress_callback(len(pkg_dict)+1, 1)
    with closing(open_addons()) as addons:
        for i, (name, (_, version)) in enumerate(pkg_dict.items()):
            if force or name not in addons or addons[name.lower()].available_version != version:
                try:
                    data = pypi.release_data(name, version)
                    rel = pypi.release_urls(name, version)[0]

                    if readthedocs:
                        try:
                            docs = readthedocs.project.get(slug=name.lower())['objects'][0]
                        except:
                            docs = {}
                    addons[name.lower()] = OrangeAddOn(name = name,
                                               available_version = data['version'],
                                               installed_version = addons[name.lower()].installed_version if name.lower() in addons else None,
                                               summary = data['summary'],
                                               description = data.get('description', ''),
                                               author = str((data.get('author', '') or '') + ' ' + (data.get('author_email', '') or '')).strip(),
                                               docs_url = data.get('docs_url', docs.get('subdomain', '')),
                                               keywords = data.get('keywords', "").split(","),
                                               homepage = data.get('home_page', ''),
                                               package_url = data.get('package_url', ''),
                                               release_url = rel.get('url', None),
                                               release_size = rel.get('size', -1),
                                               python_version = rel.get('python_version', None))
                except Exception, e:
                    import traceback
                    traceback.print_exc()
                    warnings.warn('Could not load data for the following add-on: %s'%name)
            if progress_callback:
                progress_callback(len(pkg_dict)+1, i+2)
        addons_corrupted = False

    rebuild_index()

def load_installed_addons():
    found = set()
    with closing(open_addons()) as addons:
        for entry_point in pkg_resources.iter_entry_points(ADDONS_ENTRY_POINT):
            name, version = entry_point.dist.project_name, entry_point.dist.version
            #TODO We could import setup.py from entry_point.location and load descriptions and such ...
            if name.lower() in addons:
                addons[name.lower()] = addons[name.lower()]._replace(installed_version = version)
            else:
                addons[name.lower()] = OrangeAddOn(name = name,
                    available_version = None,
                    installed_version = version,
                    summary = "",
                    description = "",
                    author = "",
                    docs_url = "",
                    keywords = "",
                    homepage = "",
                    package_url = "",
                    release_url = "",
                    release_size = None,
                    python_version = None)
            found.add(name.lower())
        for name in set(addons).difference(found):
            addons[name.lower()] = addons[name.lower()]._replace(installed_version = None)
    rebuild_index()

def run_setup(setup_script, args):
    old_dir = os.getcwd()
    save_argv = sys.argv[:]
    save_path = sys.path[:]
    setup_dir = os.path.abspath(os.path.dirname(setup_script))
    temp_dir = os.path.join(setup_dir,'temp')
    if not os.path.isdir(temp_dir): os.makedirs(temp_dir)
    save_tmp = tempfile.tempdir
    save_modules = sys.modules.copy()
    try:
        tempfile.tempdir = temp_dir
        os.chdir(setup_dir)
        try:
            sys.argv[:] = [setup_script]+list(args)
            sys.path.insert(0, setup_dir)
            execfile(
                    "setup.py",
                    {'__file__':setup_script, '__name__':'__main__'}
                )
        except SystemExit, v:
            if v.args and v.args[0]:
                raise
                # Normal exit, just return
    finally:
        sys.modules.update(save_modules)
        for key in list(sys.modules):
            if key not in save_modules: del sys.modules[key]
        os.chdir(old_dir)
        sys.path[:] = save_path
        sys.argv[:] = save_argv
        tempfile.tempdir = save_tmp


def install(name, progress_callback=None):
    if progress_callback:
        progress_callback(1, 0)
    import site
    try:
        import urllib
        rh = (lambda done, bs, fs: progress_callback(fs/bs, done)) if progress_callback else None
        with closing(open_addons()) as addons:
            egg = urllib.urlretrieve(addons[name.lower()].release_url, reporthook=rh)[0]
    except Exception, e:
        raise Exception("Unable to download add-on from repository: %s" % e)

    try:
        try:
            tmpdir = tempfile.mkdtemp()
            egg_contents = tarfile.open(egg)
            egg_contents.extractall(tmpdir)
            with closing(open_addons()) as addons:
                setup_py = os.path.join(tmpdir, name+'-'+addons[name.lower()].available_version, 'setup.py')
        except Exception, e:
            raise Exception("Unable to unpack add-on: %s" % e)

        if not os.path.isfile(setup_py):
            raise Exception("Unable to install add-on - it is not properly packed.")

        try:
            switches = []
            if site.USER_SITE in sys.path:   # we're not in a virtualenv
                switches.append('--user')
            run_setup(setup_py, ['install'] + switches)
        except Exception, e:
            raise Exception("Unable to install add-on: %s" % e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    for p in list(sys.path):
        site.addsitedir(p)
    reload(pkg_resources)
    for p in list(sys.path):
        pkg_resources.find_distributions(p)
    from orngRegistry import load_new_addons
    load_new_addons()
    load_installed_addons()
    for func in addon_refresh_callback:
        func()

def uninstall(name, progress_callback=None):
    try:
        import pip.req
        ao = pip.req.InstallRequirement(name, None)
        ao.uninstall(True)
    except ImportError:
        raise Exception("Pip is required for add-on uninstallation. Install pip and try again.")

def upgrade(name, progress_callback=None):
    install(name, progress_callback)

load_installed_addons()



# Support for loading legacy "registered" add-ons
def __read_addons_list(addons_file, systemwide):
    if os.path.isfile(addons_file):
        return [tuple([x.strip() for x in lne.split("\t")])
                for lne in file(addons_file, "rt")]
    else:
        return []

registered = __read_addons_list(os.path.join(Orange.utils.environ.orange_settings_dir, "add-ons.txt"), False) + \
             __read_addons_list(os.path.join(Orange.utils.environ.install_dir, "add-ons.txt"), True)

for name, path in registered:
    for p in [os.path.join(path, "widgets", "prototypes"),
          os.path.join(path, "widgets"),
          path,
          os.path.join(path, "lib-%s" % "-".join(( sys.platform, "x86" if (platform.machine()=="")
          else platform.machine(), ".".join(map(str, sys.version_info[:2])) )) )]:
        if os.path.isdir(p) and not any([Orange.utils.environ.samepath(p, x)
                                         for x in sys.path]):
            if p not in sys.path:
                sys.path.insert(0, p)

#TODO Show some progress to the user at least during the installation procedure.
