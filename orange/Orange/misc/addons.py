"""
Orange.misc.addons module provides a framework for Orange add-on management. As
soon as it is imported, the following initialization takes place: the list of
installed add-ons is loaded, their directories are added to python path
(:obj:`sys.path`) the callback list is initialized the stored repository list is
loaded. The most important consequence of importing the module is thus the
ability to import add-ons' modules, because they are now in the python path.

.. attribute:: available_repositories

   List of add-on repository descriptors (instances of
   :class:`OrangeAddOnRepository`).

.. attribute:: addon_directories

   List of directories that have been added to the path to make use of add-ons
   possible; see :obj:`add_addon_directories_to_path`.

.. attribute:: registered_addons

   A list of registered add-on descriptors (instances of
   :class:`OrangeRegisteredAddOn`).

.. attribute:: available_addons

   A dictionary mapping URLs of repositories to instances of
   :class:`OrangeAddOnRepository`.

.. attribute:: installed_addons

   A dictionary mapping GUIDs to instances of :class:`OrangeAddOnInstalled`.

.. autofunction:: load_installed_addons_from_dir

.. autofunction:: repository_list_filename

.. autofunction:: load_repositories

.. autofunction:: save_repositories

.. autofunction:: update_default_repositories

.. autofunction:: add_addon_directories_to_path

.. autofunction:: install_addon

.. autofunction:: install_addon_from_repo

.. autofunction:: load_addons

.. autofunction:: refresh_addons

.. autofunction:: register_addon

.. autofunction:: unregister_addon

Add-on descriptors and packaging routines
=========================================

.. autofunction:: suggest_version

.. autoclass:: OrangeRegisteredAddOn
   :members:
   :show-inheritance:

.. autoclass:: OrangeAddOn
   :members:
   :show-inheritance:

.. autoclass:: OrangeAddOnInRepo
   :members:
   :show-inheritance:

.. autoclass:: OrangeAddOnInstalled
   :members:
   :show-inheritance:

Add-on repository descriptors
=============================

.. autoclass:: OrangeAddOnRepository
   :members:
   :show-inheritance:
   
.. autoclass:: OrangeDefaultAddOnRepository
   :members:
   :show-inheritance:

Exception classes
=================

.. autoclass:: RepositoryException
   :members:
   :show-inheritance:

.. autoclass:: InstallationException
   :members:
   :show-inheritance:

.. autoclass:: PackingException
   :members:
   :show-inheritance:

"""


import xml.dom.minidom
import re
import os
import sys
import glob
import time
import socket
import urllib  # urllib because we need 'urlretrieve'
import urllib2 # urllib2 because it reports HTTP Errors for 'urlopen'
import bisect
import platform

import Orange.misc.environ
import widgetParser
from fileutil import *
from fileutil import _zip_open
from zipfile import ZipFile

import warnings

socket.setdefaulttimeout(120)  # In seconds.

class PackingException(Exception):
    """
    An exception that occurs during add-on packaging. Behaves exactly as
    :class:`Exception`.
    
    """
    pass

def suggest_version(current_version):
    """
    Automatically construct a version string of form "year.month.day[.number]". 
    If the passed "current version" is already in this format and contains
    identical date, the last number is incremented if it exists; otherwise ".1"
    is appended.
    
    :param current_version: version on which to base the new version; is used
        only in case it is in the same format.
    :type current_version: str
    
    """
    
    version = time.strftime("%Y.%m.%d")
    try:
        xmlver_int = map(int, current_version.split("."))
    except:
        xmlver_int = []
    ver_int = map(int, version.split("."))
    if xmlver_int[:3] == ver_int[:3]:
        version += ".%d" % ((xmlver_int[3] if len(xmlver_int)>3 else 0) +1)
    return version

class OrangeRegisteredAddOn():
    """
    An add-on that is not linked to an on-line repository, but resides in an
    independent directory and has been registered in Orange to be loaded when
    Canvas is run. Helper methods are also implemented to enable packaging of
    a registered add-on into an .oao package, including methods to generate
    a skeleton of documentation files.
    
    .. attribute:: id
    
       ID of the add-on. IDs of registered add-ons are in form
       "registered:<dir>", where <dir> is the directory of add-on's files.
    
    .. attribute:: name
       
       name of the add-on.
       
    .. attribute:: directory
    
       the directory where the add-on's files reside.
    
    .. attribute:: systemwide
    
       a flag indicating whether the add-on is registered system-wide, i.e.
       for all OS users.
    
    """
    
    def __init__(self, name, directory, systemwide=False):
        """
        Constructor only sets the attributes.
        
        :param name: name of the add-on.
        :type name: str
        
        :param directory: full path to the add-on's files.
        :type directory: str
        
        :param systemwide: determines whether the add-on is installed
            systemwide, ie. for all users.
        :type systemwide: boolean
        """
        self.name = name
        self.directory = directory
        self.systemwide = systemwide
        
        # Imitate real add-ons behaviour
        self.id = "registered:"+directory

    # Imitate real add-ons behaviour
    def has_single_widget(self):
        """
        Always return False: this feature is not implemented for registered
        add-ons.
        """
        return False

    def directory_documentation(self):
        """
        Return the documentation directory -- the "doc" directory under the
        add-on's directory.
        """
        return os.path.join(self.directory, "doc")

    def uninstall(self, refresh=True):
        """
        Uninstall, or rather unregister, the registered add-on. The files in
        add-on's directory are not deleted or in any other way changed.
        
        :param refresh: determines whether add-on list change callback
            functions are to be called after the unregistration process. This
            should always be True, except when multiple operations are executed
            in a batch.
        :type refresh: boolean
        """
        try:
            unregister_addon(self.name, self.directory, user_only=True)            
            if refresh:
                refresh_addons()
            return True
        except Exception, e:
            raise InstallationException("Unable to unregister add-on: %s" %
                                        (self.name, e))

    def prepare(self, id=None, name=42, version="auto", description=None,
                tags=None, author_organizations=None, author_creators=None,
                author_contributors=None, preferred_directory=None,
                homepage=None):
        """
        Prepare the add-on for packaging into an .oao ZIP file and add the
        necessary files to the add-on directory (possibly overwriting some!).

        :param id: ID of the add-on. Must be a valid GUID; None means it is
            retained from existing addon.xml if it exists, otherwise a new GUID
            is generated.
        :type id: str
        
        :param name: name of the add-on; None retains existing value if it
            exists and raises exception otherwise; the default value of 42
            uses :obj:`self.name`.
        :type name: str
            
        :param version: version of the add-on. None retains existing value if
            it exists and does the same as "auto" otherwise; "auto" generates a
            new version number from the current date in format 'yyyy.mm.dd'
            (see :obj:`Orange.misc.addons.suggest_version`); if that is equal
            to the current version, another integer component is appended.
        :type version: str
        
        :param description: add-on's description. None retains existing value
            if it exists and raises an exception otherwise.
        :type description: str
        
        :param tags: tags; None retains existing value if it exists, else
            defaults to [].
        :type tags: list of str
        
        :param author_organizations: list of authoring organizations. None
            retains existing value if it exists, else defaults to [].
        :type author_organizations: list of str
        
        :param author_creators: list of names of authors. None
            retains existing value if it exists, else defaults to [].
        :type author_creators: list of str

        :param author_contributors: list of additional organizations or people
            that have contributed to the add-on development. None
            retains existing value if it exists, else defaults to [].
        :type author_contributors: list of str

        :param preferred_directory: default directory name for installation.
            None retains existing value, "" removes the tag from the XML.
        :type preferred_directory: str
            
        :param homepage: the URL of add-on's website. None retains existing
            value, "" removes the tag from the XML.
        :type homepage: str
        """
        ##########################
        # addon.xml maintenance. #
        ##########################
        addon_xml_path = os.path.join(self.directory, "addon.xml")
        try:
            xmldoc = xml.dom.minidom.parse(addon_xml_path)
        except Exception, e:
            warnings.warn("Could not load addon.xml because \"%s\"; a new one "+
                          "will be created." % e, Warning, 0)
            impl = xml.dom.minidom.getDOMImplementation()
            xmldoc = impl.createDocument(None, "OrangeAddOn", None)
        xmldoc_root = xmldoc.documentElement
        # GUID
        if not id and not xml_text_of("id", parent=xmldoc_root):
            # GUID needs to be generated
            import uuid
            id = str(uuid.uuid1())
        if id:
            xml_set(xmldoc_root, "id", id)
        # name
        if name==42:
            name = self.name
        if name and name.strip():
            xml_set(xmldoc_root, "name", name.strip())
        elif not xml_text_of("name", parent=xmldoc_root):
            raise PackingException("'name' is a mandatory value!")
        name = xml_text_of("name", parent=xmldoc_root)
        # version
        xml_version = xml_text_of("version", parent=xmldoc_root)
        if not xml_version and not version:
            version = "auto"
        if version == "auto":
            version = suggest_version(xml_version)
        if version:
            xml_set(xmldoc_root, "version", version)
        # description
        meta = get_element_nonrecursive(xmldoc_root, "meta", create=True)
        if description and description.strip():
            xml_set(meta, "description", description.strip())
        elif not xml_text_of("description", parent=meta):
            raise PackingException("'description' is a mandatory value!")
        # tags
        def update_list(root, node_name, list):
            listNode = get_element_nonrecursive(root, node_name)
            while listNode:
                root.removeChild(listNode)
                listNode = get_element_nonrecursive(root, node_name)
            for value in list:
                root.appendChild(create_text_element(node_name, value))
        if tags!=None:
            tags_node = get_element_nonrecursive(meta, "tags", create=True)
            update_list(tags_node, "tag", tags)
        # authors
        if author_organizations!=None or author_contributors!=None or \
           author_creators!=None:
            authorsNode = get_element_nonrecursive(meta, "authors", create=True)
            if author_organizations!=None: update_list(authorsNode,
                                                       "organization",
                                                       author_organizations)
            if author_creators!=None:      update_list(authorsNode,
                                                       "creator",
                                                       author_creators)
            if author_contributors!=None:  update_list(authorsNode,
                                                       "contributor",
                                                       author_contributors)
        #  preferred_directory
        if preferred_directory != None:
            xml_set(xmldoc_root, "preferred_directory", preferred_directory
                    if preferred_directory else None)
        #  homepage
        if homepage != None:
            xml_set(xmldoc_root, "homepage", homepage if homepage else None)
            
        import codecs
        xmldoc.writexml(codecs.open(addon_xml_path, 'w', "utf-8"),
                        encoding="UTF-8")
        sys.stderr.write("Updated addon.xml written.\n")

        ##########################
        # style.css creation     #
        ##########################
        localcss = os.path.join(self.directory_documentation(), "style.css")
        orangecss = os.path.join(Orange.misc.environ.doc_install_dir, "style.css")
        if not os.path.isfile(localcss):
            if os.path.isfile(orangecss):
                import shutil
                shutil.copy(orangecss, localcss)
                sys.stderr.write("doc/style.css created.\n")
            else:
                raise PackingException("Could not find style.css in orange"+\
                                       " documentation directory.")

        ##########################
        # index.html creation    #
        ##########################
        if not os.path.isdir(self.directory_documentation()):
            os.mkdir(self.directory_documentation())
        hasIndex = False
        for fname in ["main", "index", "default"]:
            for ext in ["html", "htm"]:
                hasIndex = hasIndex or os.path.isfile(os.path.join(self.directory_documentation(),
                                                                   fname+"."+ext))
        if not hasIndex:
            indexFile = open( os.path.join(self.directory_documentation(),
                                           "index.html"), 'w')
            indexFile.write('<html><head><link rel="stylesheet" '+\
                            'href="style.css" type="text/css" /><title>%s'+\
                            '</title></head><body><h1>Module Documentation'+\
                            '</h1>%s</body></html>' % (name+" Orange Add-on "+ \
                                                       "Documentation",
                            "This is where technical add-on module "+\
                            "documentation is. Well, at least it <i>should</i>"+\
                            " be."))
            indexFile.close()
            sys.stderr.write("doc/index.html written.\n")
            
        ##########################
        # iconlist.html creation #
        ##########################
        wdocdir = os.path.join(self.directory_documentation(), "widgets")
        if not os.path.isdir(wdocdir): os.mkdir(wdocdir)
        open(os.path.join(wdocdir, "index.html"), 'w').write(self.iconlist_html())
        sys.stderr.write("Widget list (doc/widgets/index.html) written.\n")

        ##########################
        # copying the icons      #
        ##########################
        icondir = os.path.join(self.directory, "widgets", "icons")
        icondocdir = os.path.join(wdocdir, "icons")
        proticondir = os.path.join(self.directory, "widgets", "prototypes",
                                   "icons")
        proticondocdir = os.path.join(wdocdir, "prototypes", "icons")

        import shutil
        iconbg_file = os.path.join(Orange.misc.environ.icons_install_dir, "background_32.png")
        iconun_file = os.path.join(Orange.misc.environ.icons_install_dir, "Unknown.png")
        if not os.path.isdir(icondocdir): os.mkdir(icondocdir)
        if os.path.isfile(iconbg_file): shutil.copy(iconbg_file, icondocdir)
        if os.path.isfile(iconun_file): shutil.copy(iconun_file, icondocdir)
        
        if os.path.isdir(icondir):
            import distutils.dir_util
            distutils.dir_util.copy_tree(icondir, icondocdir)
        if os.path.isdir(proticondir):
            import distutils.dir_util
            if not os.path.isdir(os.path.join(wdocdir, "prototypes")):
                os.mkdir(os.path.join(wdocdir, "prototypes"))
            if not os.path.isdir(proticondocdir): os.mkdir(proticondocdir)
            distutils.dir_util.copy_tree(proticondir, proticondocdir)
        sys.stderr.write("Widget icons copied to doc/widgets/.\n")


    #####################################################
    # What follows are ugly HTML generators.            #
    #####################################################
    def widget_doc_skeleton(self, widget, prototype=False):
        """
        Return an HTML skeleton for documentation of a widget.
        
        :param widget: widget metadata.
        :type widget: :class:`widgetParser.WidgetMetaData`
        
        :param prototype: determines, whether this is a prototype widget. This
            is important to generate appropriate relative paths to the icons and
            CSS.
        :type prototype: boolean
        """
        wfile = os.path.splitext(os.path.split(widget.filename)[1])[0][2:]
        pathprefix = "../" if prototype else ""
        iconcode = '\n<p><img class="screenshot" style="z-index:2; border: none; height: 32px; width: 32px; position: relative" src="%s" title="Widget: %s" width="32" height="32" /><img class="screenshot" style="margin-left:-32px; z-index:1; border: none; height: 32px; width: 32px; position: relative" src="%sicons/background_32.png" width="32" height="32" /></p>' % (widget.icon, widget.name, pathprefix)
        
        inputscode = """<DT>(None)</DT>"""
        outputscode = """<DT>(None)</DT>"""
        il, ol = eval(widget.inputList), eval(widget.outputList)
        if il:
            inputscode = "\n".join(["<dt>%s (%s)</dt>\n<dd>Describe here, what this input does.</dd>\n" % (p[0], p[1]) for p in il])
        if ol:
            outputscode = "\n".join(["<dt>%s (%s)</dt>\n<dd>Describe here, what this output does.</dd>\n" % (p[0], p[1]) for p in ol])
        html = """<html>
<head>
<title>%s</title>
<link rel=stylesheet href="%s../style.css" type="text/css" media=screen>
</head>

<body>

<h1>%s</h1>
%s
<p>This widget does this and that..</p>

<h2>Channels</h2>

<h3>Inputs</h3>

<dl class=attributes>
%s
</dl>

<h3>Outputs</h3>
<dl class=attributes>
%s
</dl>

<h2>Description</h2>

<!-- <img class="leftscreenshot" src="%s.png" align="left"> -->

<p>This is a widget which ...</p>

<p>If you press <span class="option">Reload</span>, something will happen. <span class="option">Commit</span> button does something else.</p>

<h2>Examples</h2>

<p>This widget is used in this and that way. It often gets data from
the <a href="Another.htm">Another Widget</a>.</p>

<!-- <img class="schema" src="%s-Example.png" alt="Schema with %s widget"> -->

</body>
</html>""" % (widget.name, pathprefix, widget.name, iconcode, inputscode,
              outputscode, wfile, wfile, widget.name)
        return html
        
    
    def iconlist_html(self, create_skeleton_docs=True):
        """
        Prepare and return an HTML document, containing a table of widget icons.
        
        :param create_skeleton_docs: determines whether documentation skeleton for
            widgets without documentation should be generated (ie. whether the
            method :obj:`widget_doc_skeleton` should be called.
        :type create_skeleton_docs: boolean
        """
        html = """
<style>
div#maininner {
  padding-top: 25px;
}

div.catdiv h2 {
  border-bottom: none;
  padding-left: 20px;
  padding-top: 5px;
  font-size: 14px;
  margin-bottom: 5px;
  margin-top: 0px;
  color: #fe6612;
}

div.catdiv {
  margin-left: 10px;
  margin-right: 10px;
  margin-bottom: 20px;
  background-color: #eeeeee;
}

div.catdiv table {
  width: 98%;
  margin: 10px;
  padding-right: 20px;
}

div.catdiv table td {
  background-color: white;
/*  height: 18px;*/
  margin: 25px;
  vertical-align: center;
  border-left: solid #eeeeee 10px;
  border-bottom: solid #eeeeee 3px;
  font-size: 13px;
}

div.catdiv table td.left {
  width: 3%;
  height: 28px;
  padding: 0;
  margin: 0;
}

div.catdiv table td.left-nodoc {
  width: 3%;
  color: #aaaaaa;
  padding: 0;
  margin: 0
}


div.catdiv table td.right {
  padding-left: 5px;
  border-left: none;
  width: 22%;
  font-size: 11px;
}

div.catdiv table td.right-nodoc {
  width: 22%;
  padding-left: 5px;
  border-left: none;
  color: #aaaaaa;
  font-size: 11px;
}

div.catdiv table td.empty {
  background-color: #eeeeee;
}


.rnd1 {
 height: 1px;
 border-left: solid 3px #ffffff;
 border-right: solid 3px #ffffff;
 margin: 0px;
 padding: 0px;
}

.rnd2 {
 height: 2px;
 border-left: solid 1px #ffffff;
 border-right: solid 1px #ffffff;
 margin: 0px;
 padding: 0px;
}

.rnd11 {
 height: 1px;
 border-left: solid 1px #eeeeee;
 border-right: solid 1px #eeeeee;
 margin: 0px;
 padding: 0px;
}

.rnd1l {
 height: 1px;
 border-left: solid 1px white;
 border-right: solid 1px #eeeeee;
 margin: 0px;
 padding: 0px;
}

div.catdiv table img {
  border: none;
  height: 28px;
  width: 28px;
  position: relative;
}
</style>

<script>
function setElColors(t, id, color) {
  t.style.backgroundColor=document.getElementById('cid'+id).style.backgroundColor = color;
}
</script>

<p style="font-size: 16px; font-weight: bold">Catalog of widgets</p>
        """
        wdir = os.path.join(self.directory, "widgets")
        pdir = os.path.join(wdir, "prototypes")
        widgets = {}
        for (prototype, filename) in [(False, filename) for filename in
                                      glob.iglob(os.path.join(wdir, "*.py"))] +\
                                     [(True, filename) for filename in
                                      glob.iglob(os.path.join(pdir, "*.py"))]:
            if os.path.isdir(filename) or os.path.islink(filename):
                continue
            try:
                meta = widgetParser.WidgetMetaData(file(filename).read(),
                                                   "Prototypes" if prototype else "Uncategorized",
                                                   enforceDefaultCategory=prototype,
                                                   filename=filename)
            except:
                continue # Probably not an Orange Widget module; skip this file.
            if meta.category in widgets:
                widgets[meta.category].append((prototype, meta))
            else:
                widgets[meta.category] = [(prototype, meta)]
        category_list = [cat for cat in widgets.keys()
                         if cat not in ["Prototypes", "Uncategorized"]]
        category_list.sort()
        for cat in ["Uncategorized"] + category_list + ["Prototypes"]:
            if cat not in widgets:
                continue
            html += """    <div class="catdiv">
    <div class="rnd1"></div>
    <div class="rnd2"></div>

    <h2>%s</h2>
    <table><tr>
""" % cat
            for i, (p, w) in enumerate(widgets[cat]):
                if (i>0) and (i%4 == 0):
                    html += "</tr><tr>\n"
                wreldir = os.path.relpath(os.path.split(w.filename)[0], wdir)\
                          if "relpath" in os.path.__dict__ else\
                          os.path.split(w.filename)[0].replace(wdir, "")
                docfile = os.path.join(wreldir,
                                       os.path.splitext(os.path.split(w.filename)[1][2:])[0] + ".htm")
                
                iconfile = os.path.join(wreldir, w.icon)
                if not os.path.isfile(os.path.join(wdir, iconfile)):
                    iconfile = "icons/Unknown.png"
                if os.path.isfile(os.path.join(self.directory_documentation(),
                                               "widgets", docfile)):
                    html += """<td id="cid%d" class="left"
      onmouseover="this.style.backgroundColor='#fff7df'"
      onmouseout="this.style.backgroundColor=null"
      onclick="this.style.backgroundColor=null; window.location='%s'">
      <div class="rnd11"></div>
      <img style="z-index:2" src="%s" title="Widget: Text File" width="28" height="28" /><img style="margin-left:-28px; z-index:1" src="icons/background_32.png" width="28" height="28" />
      <div class="rnd11"></div>
  </td>

  <td class="right"
    onmouseover="setElColors(this, %d, '#fff7df')"
    onmouseout="setElColors(this, %d, null)"
    onclick="setElColors(this, %d, null); window.location='%s'">
      %s
</td>
""" % (i, docfile, iconfile, i, i, i, docfile, w.name)
                else:
                    skeleton_filename = os.path.join(self.directory_documentation(),
                                                     "widgets",
                                                     docfile+".skeleton")
                    if not os.path.isdir(os.path.dirname(skeleton_filename)):
                        os.mkdir(os.path.dirname(skeleton_filename))
                    open(skeleton_filename, 'w').write(self.widget_doc_skeleton(w, prototype=p))
                    html += """  <td id="cid%d" class="left-nodoc">
      <div class="rnd11"></div>
      <img style="z-index:2" src="%s" title="Widget: Text File" width="28" height="28" /><img style="margin-left:-28px; z-index:1" src="icons/background_32.png" width="28" height="28" />
      <div class="rnd11"></div>
  </td>
  <td class="right-nodoc">
      <div class="rnd1l"></div>
      %s
      <div class="rnd1l"></div>

  </td>
""" % (i, iconfile, w.name)
            html += '</tr></table>\n<div class="rnd2"></div>\n<div class="rnd1"></div>\n</div>\n'
        return html
    ###########################################################################
    # Here end the ugly HTML generators. Only beautiful code from now on! ;) #
    ###########################################################################
        

class OrangeAddOn():
    """
    Stores data about an add-on for Orange. 

    .. attribute:: id
    
       ID of the add-on. IDs of registered add-ons are in form
       "registered:<dir>", where <dir> is the directory of add-on's files.
    
    .. attribute:: name
       
       name of the add-on.
       
    .. attribute:: architecture
    
       add-on structure version; currently it must have a value of 1.
    
    .. attribute:: homepage
    
       URL of add-on's web site.
       
    .. attribute:: version_str
       
       string representation of add-on's version; must be a period-separated
       list of integers.
       
    .. attribute:: version
    
       parsed value of the :obj:`version_str` attribute - a list of integers.
    
    .. attribute:: description
    
       textual description of the add-on.
       
    .. attribute:: tags
    
       textual tags that describe the add-on - a list of strings.
    
    .. attribute:: author_organizations
    
       a list of strings with names of organizations that developed the add-on.

    .. attribute:: author_creators
    
       a list of strings with names of individuals (persons) that developed the
       add-on.

    .. attribute:: author_contributors
    
       a list of strings with names of organizations and individuals (persons)
       that have made minor contributions to the add-on.
    
    .. attribute:: preferred_directory
    
       preferred name of the subdirectory under which the add-on is to be
       installed. It is not guaranteed this directory name will be used; for
       example, when such a directory already exists, another name will be
       generated during installation.
    """

    def __init__(self, xmlfile=None):
        """
        Initialize an empty add-on descriptor. Initializes attributes with data
        from an optionally passed XML add-on descriptor; otherwise sets all
        attributes to None or, in case of list attributes, an empty list.
        
        :param xmlfile: an optional file name or an instance of minidom's
            Element with XML add-on descriptor.
        :type xmlfile: :class:`xml.dom.minidom.Element` or str or
            :class:`NoneType`
        """
        self.name = None
        self.architecture = None
        self.homepage = None
        self.id = None
        self.version_str = None
        self.version = None
        
        self.description = None
        self.tags = []
        self.author_organizations = []
        self.author_creators = []
        self.author_contributors = []
        
        self.preferred_directory = None
        
        self.widgets = []  # List of widgetParser.WidgetMetaData objects
        
        if xmlfile:
            xml_doc_root = xmlfile if xmlfile.__class__ is xml.dom.minidom.Element else\
                         xml.dom.minidom.parse(xmlfile).documentElement
            try:
                self.parsexml(xml_doc_root)
            finally:
                xml_doc_root.unlink()

    def clone(self, new=None):
        """
        Clone the add-on descriptor, effectively making a deep copy.
        
        :param new: a new instance of this class into which to copy the values
            of attributes; if None, a new instance is constructed.
        :type new: :class:`OrangeAddOn` or :class:`NoneType`
        """
        if not new:
            new = OrangeAddOn()
        new.name = self.name
        new.architecture = self.architecture
        new.homepage = self.homepage
        new.id = self.id
        new.version_str = self.version_str
        new.version = list(self.version)
        new.description = self.description
        new.tags = list(self.tags)
        new.author_organizations = list(self.author_organizations)
        new.author_creator = list(self.author_creators)
        new.author_contributors = list(self.author_contributors)
        new.prefferedDirectory = self.preferred_directory
        new.widgets = [w.clone() for w in self.widgets]
        return new

    def directory_documentation(self):
        """
        Return the documentation directory -- the "doc" directory under the
        add-on's directory.
        """
        #TODO This might be redefined in orngConfiguration.
        return os.path.join(self.directory, "doc")

    def parsexml(self, root):
        """
        Parse the add-on's XML descriptor and set object's attributes
        accordingly.
        
        :param root: root of the add-on's descriptor (the node with tag name
            "OrangeAddOn").
        :type root: :class:`xml.dom.minidom.Element`
        """
        if root.tagName != "OrangeAddOn":
            raise Exception("Invalid XML add-on descriptor: wrong root element name!")
        
        mandatory = ["id", "architecture", "name", "version", "meta"]
        textnodes = {"id": "id", "architecture": "architecture", "name": "name",
                     "version": "version_str", 
                     "preferredDirectory": "preferredDirectory",
                     "homePage": "homepage"}
        for node in [n for n in root.childNodes if n.nodeType==n.ELEMENT_NODE]:
            if node.tagName in mandatory:
                mandatory.remove(node.tagName)
                
            if node.tagName in textnodes:
                setattr(self, textnodes[node.tagName],
                        widgetParser.xml_text_of(node))
            elif node.tagName == "meta":
                for node in [n for n in node.childNodes
                             if n.nodeType==n.ELEMENT_NODE]:
                    if node.tagName == "description":
                        self.description = widgetParser.xml_text_of(node, True)
                    elif node.tagName == "tags":
                        for tagNode in [n for n in node.childNodes
                                        if n.nodeType==n.ELEMENT_NODE and
                                        n.tagName == "tag"]:
                            self.tags.append(widgetParser.xml_text_of(tagNode))
                    elif node.tagName == "authors":
                        authorTypes = {"organization": self.author_organizations,
                                       "creator": self.author_creators,
                                       "contributor": self.author_contributors}
                        for authorNode in [n for n in node.childNodes
                                           if n.nodeType==n.ELEMENT_NODE and
                                           n.tagName in authorTypes]:
                            authorTypes[authorNode.tagName].append(widgetParser.xml_text_of(authorNode))
            elif node.tagName == "widgets":
                for node in [n for n in node.childNodes
                             if n.nodeType==n.ELEMENT_NODE]:
                    if node.tagName == "widget":
                        self.widgets.append(widgetParser.WidgetMetaData(node))
        
        if "afterparse" in self.__class__.__dict__:
            self.afterparse(root)
        
        self.validate_architecture()
        if mandatory:
            raise Exception("Mandatory elements missing: "+", ".join(mandatory))
        self.validate_id()
        self.validate_name()
        self.validate_version()
        self.validate_description()
        if self.preferred_directory==None:
            self.preferred_directory = self.name

    def validate_architecture(self):
        """
        Raise an exception if the :obj:`architecture` (structure of the add-on)
        is not supported. Currently, only architecture 1 exists.
        """
        if self.architecture != "1":
            raise Exception("Only architecture '1' is supported by current Orange!")
    
    def validate_id(self):
        """
        Raise an exception if the :obj:`id` is not a valid GUID.
        """
        idPattern = re.compile("[0-9a-z]{8}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{12}")
        if not idPattern.match(self.id):
            raise Exception("Invalid ID!")

    def validate_name(self):
        """
        Raise an exception if the :obj:`name` is empty (or contains only
        whitespace).
        """
        if self.name.strip() == "":
            raise Exception("Name is a mandatory field!")
    
    def validate_version(self):
        """
        Parse the :obj:`version_str` and populate the :obj:`version` attribute.
        Raise an exception if the version is not in correct format (ie. a
        period-separated list of integers).
        """
        self.version = []  
        for sub in self.version_str.split("."):
            try:
                self.version.append(int(sub))
            except:
                self.version = []
                raise Exception("Invalid version string: '%s' is not an integer!" % sub)
        self.version_str = ".".join(map(str,self.version))
            
    def validate_description(self):
        """
        Raise an exception if the :obj:`description` is empty (or contains only
        whitespace).
        """
        if self.name.strip() == "":
            raise Exception("Description is a mandatory field!")
        
    def has_single_widget(self):
        """
        Determine whether the add-on contains less than two widgets.
        """
        return len(self.widgets) < 2
        

class OrangeAddOnInRepo(OrangeAddOn):
    """
    Stores data about an add-on for Orange that exists in a repository.
    Additional attributes are:
    
    .. attribute:: repository
    
    A repository object (instance of :class:`OrangeAddOnRepository`) that
    contains data about the add-on's repository.

    .. attribute:: filename
    
    The name of .oao file in repository.
    
    """
     
    def __init__(self, repository, filename=None, xmlfile=None):
        """
        Constructor only sets the attributes.
        
        :param repository: the repository that contains the add-on.
        :type repostitory: :class:`OrangeAddOnRepository`
        
        :param filename: name of the .oao file in repository (is used only if
            the XML file does not specify the filename).
        :type filename: str
        
        :param xmlfile: an optional file name or an instance of minidom's
            Element with XML add-on descriptor.
        :type xmlfile: :class:`xml.dom.minidom.Element` or str or
            :class:`NoneType`
        """
        OrangeAddOn.__init__(self, xmlfile)
        self.repository = repository
        if "filename" not in self.__dict__:
            self.filename = filename
    
    def afterparse(self, xml_root):  # Called by OrangeAddOn.parsexml()
        """
        Read the filename attribute from the XML. This method is called by
        :obj:`OrangeAddOn.parsexml`.
        """
        if xml_root.hasAttribute("filename"):
            self.filename = xml_root.getAttribute("filename")
            
    def clone(self, new=None):
        """
        Clone the add-on descriptor, effectively making a deep copy.
        
        :param new: a new instance of this class into which to copy the values
            of attributes; if None, a new instance is constructed.
        :type new: :class:`OrangeAddOn` or :class:`NoneType`
        """
        if not new:
            new = OrangeAddOnInRepo(self.repository)
        new.filename = self.filename
        return OrangeAddOn.clone(self, new)

class OrangeAddOnInstalled(OrangeAddOn):
    """
    Stores data about an add-on for Orange that has been installed from a
    repository. Additional attribute is:
    
    .. attribute:: directory
    
    Directory of add-on's files.
    
    """
    def __init__(self, directory):
        """
        Constructor only sets the attributes.
        
        :param directory: directory of add-on's files, including an XML
            descriptor to read.
        :type directory: str
        """
        OrangeAddOn.__init__(self, os.path.join(directory, "addon.xml")
                             if directory else None)
        self.directory = directory
    
    def uninstall(self, refresh=True):
        """
        Uninstall the installed add-on. WARNING: all files in add-on's directory
        are deleted!
        
        :param refresh:  determines whether add-on list change callback
            functions are to be called after the unregistration process. This
            should always be True, except when multiple operations are executed
            in a batch.
        :type refresh: boolean
        """
        try:
            _deltree(self.directory)
            del installed_addons[self.id]
            if refresh:
                refresh_addons()
            return True
        except Exception, e:
            raise InstallationException("Unable to remove add-on: %s" %
                                        (self.name, e))
        
    def clone(self, new=None):
        """
        Clone the add-on descriptor, effectively making a deep copy.
        
        :param new: a new instance of this class into which to copy the values
            of attributes; if None, a new instance is constructed.
        :type new: :class:`OrangeAddOn` or :class:`NoneType`
        """
        if not new:
            new = OrangeAddOnInstalled(None)
        new.directory = self.directory
        return OrangeAddOn.clone(self, new)
        
available_addons = {}  # RepositoryURL -> OrangeAddOnRepository object 
installed_addons = {}  # ID -> OrangeAddOnInstalled object
registered_addons = [] # OrangeRegisteredAddOn objects

class RepositoryException(Exception):
    """
    An exception that occurs during access to repository location. Behaves
    exactly as :class:`Exception`.

    """
    pass

global index_re
index_re = "[^a-z0-9-']"  # RE for splitting entries in the search index

class OrangeAddOnRepository:
    """
    Repository of Orange add-ons.
    
    .. attribute:: name
    
    A local descriptive name for the repository.
    
    .. attribute:: url
    
    URL of the repository root; http and file protocols are supported.
    
    .. attribute:: addons
    
    A dictionary mapping GUIDs to lists of add-on objects (of class
    :class:`OrangeAddOnInRepo`). Each GUID is thus mapped to at least one,
    but possibly more, different versions of add-on.
    
    .. attribute:: index
    
    A search index: sorted list of tuples (s, GUID), where such an entry
    signifies that when searching for a string that s starts with, add-on with
    the given GUID should be among results.
    
    .. attribute:: last_refresh_utc
    
    :obj:`time.time` of the last reloading of add-on list.
    
    .. attribute:: has_web_script
    
    A boolean indicating whether this is an http repository that contains the
    appropriate server-side python script that returns an XML with a list of
    add-ons.
    
    """
    
    def __init__(self, name, url, load=True, force=False):
        """
        :param name: a local descriptive name for the repository.
        :type name: str
        
        :param url: URL of the repository root; http and file protocols are
            supported. If the protocol is not given, file:// is assumed.
        :type url: str
        
        :param load: determines whether the list of repository's add-ons should
            be loaded immediately.
        :type load: boolean
        
        :param force: determines whether loading of repository's add-on list
            is mandatory, ie. if an exception is to be raised in case of
            connection failure.
        :type force: boolean
        """
        
        self.name = name
        self.url = url
        self.checkurl()
        self.addons = {}
        self.index = []
        self.last_refresh_utc = 0
        self._refresh_index()
        self.has_web_script = False
        if load:
            try:
                self.refreshdata(True, True)
            except Exception, e:
                if force:
                    warnings.warn("Couldn't load data from repository '%s': %s"
                                  % (self.name, e), Warning, 0)
                    return
                raise e
        
    def clone(self, new=None):
        """
        Clone the repository descriptor, effectively making a deep copy.
        
        :param new: a new instance of this class into which to copy the values
            of attributes; if None, a new instance is constructed.
        :type new: :class:`OrangeAddOnRepository` or :class:`NoneType`
        """
        if not new:
            new = OrangeAddOnRepository(self.name, self.url, load=False)
        new.addons = {}
        for (id, versions) in self.addons.items():
            new.addons[id] = [ao.clone() for ao in versions]
        new.index = list(self.index)
        new.last_refresh_utc = self.last_refresh_utc
        new.has_web_script = self.has_web_script if hasattr(self, 'has_web_script') else False
        return new

    def checkurl(self):
        """
        Check the URL for validity. Return True if it begins with "file://" or
        "http://" or if it does not specify a protocol (in this case, file:// is
        assumed).
        """
        supportedProtocols = ["file", "http"]
        if "://" not in self.url:
            self.url = "file://"+self.url
        protocol = self.url.split("://")[0]
        if protocol not in supportedProtocols:
            raise Exception("Unable to load repository data: protocol '%s' not supported!" %
                            protocol)

    def _add_addon(self, addon):
        """
        Add the given addon descriptor to the :obj:`addons` dictionary.
        Operation is sucessful only if there is no add-on with equal GUID
        (:obj:`OrangeAddOn.id`) and version
        (:obj:`OrangeAddOn.version`) already in this repository.
        
        :param addon: add-on descriptor to add.
        :type addon: :class:`OrangeAddOnInRepo`
        """
        if addon.id in self.addons:
            versions = self.addons[addon.id]
            for version in versions:
                if version.version == addon.version:
                    warnings.warn("Ignoring the second occurence of addon '%s'"+
                                  ", version '%s'." % (addon.name,
                                                       addon.version_str),
                                  Warning, 0)
                    return
            versions.append(addon)
        else:
            self.addons[addon.id] = [addon]

    def _add_packed_addon(self, oaofile, filename=None):
        """
        Given a local path to an .oao file, add the addon descriptor to the
        :obj:`addons` dictionary. Specifically, "addon.xml" manifest is unpacked
        from the .oao, an :class:`OrangeAddOnInRepo` instance is constructed
        and :obj:`_add_addon` is invoked.
        
        :param oaofile: path to the .oao file.
        :type oaofile: str
        
        :param filename: name of the .oao file within the repository.
        :type filename: str
        """
        pack = ZipFile(oaofile, 'r')
        try:
            manifestfile = _zip_open(pack, 'addon.xml')
            manifest = xml.dom.minidom.parse(manifestfile).documentElement
            manifest.appendChild(widgetParser.widgets_xml(pack))
            addon = OrangeAddOnInRepo(self, filename, xmlfile=manifest)
            self._add_addon(addon)
        except Exception, e:
            raise Exception("Unable to load add-on descriptor: %s" % e)
    
    def refreshdata(self, force=False, firstload=False, interval=3600*24):
        """
        Refresh the add-on list if necessary. For an http repository, the
        server-side python script is invoked. If that fails, or if the
        repository is on local filesystem (file://), all .oao files are
        downloaded, unpacked and their manifests (addon.xml) are parsed.
        
        :param force: force a refresh, even if less than a preset amount of
            time (see parameter :obj:`interval`) has passed since last refresh
            (see attribute :obj:`last_refresh_utc`).
        :type force: boolean
        
        :param firstload: determines, whether this is the first loading of
            repository's contents. Right now, the only difference is that when
            there is no server-side repository script on an http repository and
            there are also no .oao files, this results in an exception if
            this parameter is set to True, and in a warning otherwise.
        :type firstload: boolean
        
        :parameter interval: an amount of time in seconds that must pass since
            last refresh (:obj:`last_refresh_utc`) to make the refresh happen.
        :type interval: int
        """
        if force or (self.last_refresh_utc < time.time() - interval):
            self.last_refresh_utc = time.time()
            self.has_web_script = False
            try:
                protocol = self.url.split("://")[0]
                if protocol == "http": # A remote repository
                    # Try to invoke a server-side script to retrieve add-on index (and therefore avoid downloading archives)
                    repositoryXmlDoc = None
                    try:
                        repositoryXmlDoc = urllib2.urlopen(self.url+"/addOnServer.py?machine=1")
                        repositoryXml = xml.dom.minidom.parse(repositoryXmlDoc).documentElement
                        if repositoryXml.tagName != "OrangeAddOnRepository":
                            raise Exception("Invalid XML add-on repository descriptor: wrong root element name!")
                        self.addons = {}
                        for (i, node) in enumerate([n for n
                                                    in repositoryXml.childNodes
                                                    if n.nodeType==n.ELEMENT_NODE]):
                            if node.tagName == "OrangeAddOn":
                                try:
                                    addon = OrangeAddOnInRepo(self, xmlfile=node)
                                    self._add_addon(addon)
                                except Exception, e:
                                    warnings.warn("Ignoring node nr. %d in "+
                                                  "repository '%s' because of"+
                                                  " an error: %s" % (i+1,
                                                                     self.name,
                                                                     e),
                                                  Warning, 0)
                        self.has_web_script = True
                        return True
                    except Exception, e:
                        warnings.warn("A problem occurred using server-side script on repository '%s': %s.\nAll add-ons need to be downloaded for their metadata to be extracted!"
                                      % (self.name, str(e)), Warning, 0)

                    # Invoking script failed - trying to get and parse a directory listing
                    try:
                        repoconn = urllib2.urlopen(self.url+'abc')
                        response = "".join(repoconn.readlines())
                    except Exception, e:
                        raise RepositoryException("Unable to load repository data: %s" % e)
                    addOnFiles = map(lambda x: x.split('"')[1],
                                     re.findall(r'href\s*=\s*"[^"/?]*\.oao"',
                                                response))
                    if len(addOnFiles)==0:
                        if firstload:
                            raise RepositoryException("Unable to load reposito"+
                                                      "ry data: this is not an"+
                                                      " Orange add-on "+
                                                      "repository!")
                        else:
                            warnings.warn("Repository '%s' is empty ..." %
                                          self.name, Warning, 0)
                    self.addons = {}
                    for addOnFile in addOnFiles:
                        try:
                            addOnTmpFile = urllib.urlretrieve(self.url+"/"+addOnFile)[0]
                            self._add_packed_addon(addOnTmpFile, addOnFile)
                        except Exception, e:
                            warnings.warn("Ignoring '%s' in repository '%s' "+
                                          "because of an error: %s" %
                                          (addOnFile, self.name, e),
                                          Warning, 0)
                elif protocol == "file": # A local repository: open each and every archive to obtain data
                    dir = self.url.replace("file://","")
                    if not os.path.isdir(dir):
                        raise RepositoryException("Repository '%s' is not valid: '%s' is not a directory." % (self.name, dir))
                    self.addons = {}
                    for addOnFile in glob.glob(os.path.join(dir, "*.oao")):
                        try:
                            self._add_packed_addon(addOnFile,
                                                  os.path.split(addOnFile)[1])
                        except Exception, e:
                            warnings.warn("Ignoring '%s' in repository '%s' "+
                                          "because of an error: %s" %
                                          (addOnFile, self.name, e),
                                          Warning, 0)
                return True
            finally:
                self._refresh_index()
        return False
        
    def _add_to_index(self, addon, text):
        """
        Add the words, found in given text, to the search index, to be
        associated with given add-on.
        
        :param addon: add-on to add to the search index.
        :type addon: :class:`OrangeAddOnInRepo`
        
        :param text: text from which to extract words to be added to the index.
        :type text: str
        """
        words = [word for word in re.split(index_re, text.lower())
                 if len(word)>1]
        for word in words:
            bisect.insort_right(self.index, (word, addon.id) )
                
    def _refresh_index(self):
        """
        Rebuild the search index.
        """
        self.index = []
        for addOnVersions in self.addons.values():
            for addOn in addOnVersions:
                for str in [addOn.name, addOn.description] + addOn.author_creators + addOn.author_contributors + addOn.author_organizations + addOn.tags +\
                           [" ".join([w.name, w.contact, w.description, w.category, w.tags]) for w in addOn.widgets]:
                    self._add_to_index(addOn, str)
        self.last_search_phrase = None
        self.last_search_result = None
                    
    def search_index(self, phrase):
        """
        Search the word index for the given phrase and return a list of
        matching add-ons' GUIDs. The given phrase is split into sequences
        of alphanumeric characters, just like strings are split when
        building the index, and resulting add-ons match all of the words in
        the phrase.
        
        :param phrase: a phrase to search.
        :type phrase: str
        """
        if phrase == self.last_search_phrase:
            return self.last_search_result
        
        words = [word for word in re.split(index_re, phrase.lower()) if word!=""]
        result = set(self.addons.keys())
        for word in words:
            subset = set()
            i = bisect.bisect_left(self.index, (word, ""))
            while self.index[i][0][:len(word)] == word:
                subset.add(self.index[i][1])
                i += 1
                if i>= len(self.index): break
            result = result.intersection(subset)
        self.last_search_phrase = phrase
        self.last_search_result = result
        return result
        
class OrangeDefaultAddOnRepository(OrangeAddOnRepository):
    """
    Repository of Orange add-ons that is added by default.
    
    It has a hard-coded name of "Default Orange Repository (orange.biolab.si)"
    and URL "http://orange.biolab.si/add-ons/"; those arguments cannot be
    passed to the constructor. Also, the :obj:`force` parameter is set to
    :obj:`True`. Other parameters are passed to the superclass' constructor.
    """
    
    def __init__(self, **args):
        OrangeAddOnRepository.__init__(self, "Default Orange Repository (orange.biolab.si)",
                                       "http://orange.biolab.si/add-ons/",
                                       force=True, **args)
        
    def clone(self, new=None):
        if not new:
            new = OrangeDefaultAddOnRepository(load=False)
        new.name = self.name
        new.url = self.url
        return OrangeAddOnRepository.clone(self, new)
        
def load_installed_addons_from_dir(dir):
    """
    Populate the :obj:`installed_addons` dictionary with add-ons, installed
    into direct subdirectories of the given directory.
    
    :param dir: directory to search for add-ons.
    :type dir: str
    """
    if os.path.isdir(dir):
        for name in os.listdir(dir):
            addOnDir = os.path.join(dir, name)
            if not os.path.isdir(addOnDir):
                continue
            try:
                addOn = OrangeAddOnInstalled(addOnDir)
            except Exception, e:
                warnings.warn("Add-on in directory '%s' has no valid descriptor (addon.xml): %s" % (addOnDir, e), Warning, 0)
                continue
            if addOn.id in installed_addons:
                warnings.warn("Add-on in directory '%s' has the same ID as the addon in '%s'!" % (addOnDir, installed_addons[addOn.id].directory), Warning, 0)
                continue
            installed_addons[addOn.id] = addOn

def repository_list_filename():
    """
    Return the full filename of pickled add-on repository list. It resides
    within Canvas settings directory. 
    """
    canvasSettingsDir = os.path.realpath(Orange.misc.environ.canvas_settings_dir)
    listFileName = os.path.join(canvasSettingsDir, "repositoryList.pickle")
    return listFileName

available_repositories = None
            
def load_repositories(refresh=True):
    """
    Populate the :obj:`available_repositories` list by reading the pickled
    repository list and adding the default repository
    (http://orange.biolab.si/addons) if it is not yet on the list. Optionally,
    lists of add-ons in repositories are refreshed.
    
    :param refresh: determines whether the add-on lists of repositories should
        be refreshed.
    :type refresh: boolean
    """
    listFileName = repository_list_filename()
    global available_repositories
    available_repositories = []
    if os.path.isfile(listFileName):
        try:
            import cPickle
            file = open(listFileName, 'rb')
            available_repositories = [repo.clone() for repo
                                      in cPickle.load(file)]
            file.close()
        except Exception, e:
            warnings.warn("Unable to load repository list! Error: %s" % e, Warning, 0)
    try:
        update_default_repositories(refresh=refresh)
    except Exception, e:
        warnings.warn("Unable to refresh default repositories: %s" % (e), Warning, 0)

    if refresh:
        for r in available_repositories:
            #TODO: # Should show some progress (and enable cancellation)
            try:
                r.refreshdata(force=False)
            except Exception, e:
                warnings.warn("Unable to refresh repository %s! Error: %s" % (r.name, e), Warning, 0)
    save_repositories()

def save_repositories():
    """
    Save the add-on repository list (:obj:`available_repositories`) to a 
    specific file (see :obj:`repository_list_filename`).
    """
    listFileName = repository_list_filename()
    try:
        import cPickle
        global available_repositories
        cPickle.dump(available_repositories, open(listFileName, 'wb'))
    except Exception, e:
        warnings.warn("Unable to save repository list! Error: %s" % e, Warning, 0)
    

def update_default_repositories(refresh=True):
    """
    Make sure the appropriate default repository (and no other
    :class:`OrangeDefaultAddOnRepository`) is in :obj:`available_repositories`.
    This function is called by :obj:`load_repositories`.
    
    :param refresh: determines whether the add-on list of added default
        repository should be refreshed.
    :type refresh: boolean
    """
    global available_repositories
    default = [OrangeDefaultAddOnRepository(load=False)]
    defaultKeys = [(repo.url, repo.name) for repo in default]
    existingKeys = [(repo.url, repo.name) for repo in available_repositories]
    
    for i, key in enumerate(defaultKeys):
        if key not in existingKeys:
            available_repositories.append(default[i])
            if refresh:
                default[i].refreshdata(firstload=True)
    
    to_remove = []
    for i, key in enumerate(existingKeys):
        if isinstance(available_repositories[i], OrangeDefaultAddOnRepository) and \
           key not in defaultKeys:
            to_remove.append(available_repositories[i])
    for tr in to_remove:
        available_repositories.remove(tr)

addon_directories = []
def add_addon_directories_to_path():
    """
    Add directories, related to installed add-ons, to python path, if they are
    not yet there. Added directories are also stored into
    :obj:`addon_directories`. If this function is called more than once, the
    non-first invocation first removes the entries in :obj:`addon_directories`
    from the path.
    
    If an add-on is installed in directory D, those directories are added to
    python path (:obj:`sys.path`):
   
      - D,
      - D/widgets
      - D/widgets/prototypes
      - D/lib-<platform>
      
   Here, <platform> is a "-"-separated concatenation of :obj:`sys.platform`,
   result of :obj:`platform.machine` (an empty string is replaced by "x86") and
   comma-separated first two components of :obj:`sys.version_info`.
   """
    import os, sys
    global addon_directories, registered_addons
    sys.path = [dir for dir in sys.path if dir not in addon_directories]
    for addOn in installed_addons.values() + registered_addons:
        path = addOn.directory
        for p in [os.path.join(path, "widgets", "prototypes"),
                  os.path.join(path, "widgets"),
                  path,
                  os.path.join(path, "lib-%s" % "-".join(( sys.platform, "x86"
                                                           if (platform.machine()=="")
                                                           else platform.machine(),
                                                           ".".join(map(str, sys.version_info[:2])) )) )]:
            if os.path.isdir(p) and not any([Orange.misc.environ.samepath(p, x)
                                             for x in sys.path]):
                if p not in sys.path:
                    addon_directories.append(p)
                    sys.path.insert(0, p)

def _deltree(dirname):
     if os.path.exists(dirname):
        for root,dirs,files in os.walk(dirname):
                for dir in dirs:
                        _deltree(os.path.join(root,dir))
                for file in files:
                        os.remove(os.path.join(root,file))     
        os.rmdir(dirname)

class InstallationException(Exception):
    """
    An exception that occurs during add-on installation. Behaves exactly as
    :class:`Exception`.

    """
    pass

def install_addon(oaofile, global_install=False, refresh=True):
    """
    Install an add-on from given .oao package. Installation means unpacking the
    .oao file to an appropriate directory (:obj:`Orange.misc.environ.add_ons_dir_user` or
    :obj:`Orange.misc.environ.add_ons_dir_sys`, depending on the
    :obj:`global_install` parameter), creating an
    :class:`OrangeAddOnInstalled` instance and adding this object into the
    :obj:`installed_addons` dictionary.
    
    :param global_install: determines whether the given add-on is to be
        installed globally, ie. for all users. Administrative privileges on
        the file system are usually needed for that.
    :type global_install: boolean
    
    :param refresh: determines whether add-on list change callback
        functions are to be called after the installation process. This
        should always be True, except when multiple operations are executed
        in a batch.
    :type refresh: boolean
    """
    try:
        pack = ZipFile(oaofile, 'r')
    except Exception, e:
        raise Exception("Unable to unpack the add-on '%s': %s" % (oaofile, e))
        
    try:
        for filename in pack.namelist():
            if filename[0]=="\\" or filename[0]=="/" or filename[:2]=="..":
                raise InstallationException("Refusing to install unsafe package: it contains file named '%s'!" % filename)
        
        root = Orange.misc.environ.add_ons_dir if global_install else Orange.misc.environ.add_ons_dir_user
        
        try:
            manifest = _zip_open(pack, 'addon.xml')
            addon = OrangeAddOn(manifest)
        except Exception, e:
            raise Exception("Unable to load add-on descriptor: %s" % e)
        
        if addon.id in installed_addons:
            raise InstallationException("An add-on with this ID is already installed!")
        
        # Find appropriate directory name for the new add-on.
        i = 1
        while True:
            addon_dir = os.path.join(root,
                                     addon.preferred_directory + ("" if i<2 else " (%d)"%i))
            if not os.path.exists(addon_dir):
                break
            i += 1
            if i>1000:  # Avoid infinite loop if something goes wrong.
                raise InstallationException("Cannot find an appropriate directory name for the new add-on.")
        
        # Install (unpack) add-on.
        try:
            os.makedirs(addon_dir)
        except OSError, e:
            if e.errno==13:  # Permission Denied
                raise InstallationException("No write permission for the add-ons directory!")
        except Exception, e:
                raise Exception("Cannot create a new add-on directory: %s" % e)

        try:
            if hasattr(pack, "extractall"):
                pack.extractall(addon_dir)
            else: # Python 2.5
                import shutil
                for filename in pack.namelist():
                    # don't include leading "/" from file name if present
                    if filename[0] == '/':
                        targetpath = os.path.join(addon_dir, filename[1:])
                    else:
                        targetpath = os.path.join(addon_dir, filename)
                    upperdirs = os.path.dirname(targetpath)
                    if upperdirs and not os.path.exists(upperdirs):
                        os.makedirs(upperdirs)
            
                    if filename[-1] == '/':
                        if not os.path.isdir(targetpath):
                            os.mkdir(targetpath)
                        continue
            
                    source = _zip_open(pack, filename)
                    target = file(targetpath, "wb")
                    shutil.copyfileobj(source, target)
                    source.close()
                    target.close()

            addon = OrangeAddOnInstalled(addon_dir)
            installed_addons[addon.id] = addon
        except Exception, e:
            try:
                _deltree(addon_dir)
            except:
                pass
            raise Exception("Cannot install add-on: %s"%e)
        
        if refresh:
            refresh_addons()
    finally:
        pack.close()

def install_addon_from_repo(addon_in_repo, global_install=False, refresh=True):
    """
    Retrieve the .oao file from the repository, then call :obj:`install_addon`
    on the resulting file, passing it given parameters.
    
    :param addon_in_repo: add-on in repository to be installed.
    :type addon_in_repo: :class:`OrangeAddOnInRepo`
    """
    try:
        tmpfile = urllib.urlretrieve(addon_in_repo.repository.url+"/"+addon_in_repo.filename)[0]
    except Exception, e:
        raise InstallationException("Unable to download add-on from repository: %s" % e)
    install_addon(tmpfile, global_install, refresh)

def load_addons():
    """
    Call :obj:`load_installed_addons_from_dir` on a system-wide add-on
    installation directory (:obj:`orngEnviron.addOnsDirSys`) and user-specific
    add-on installation directory (:obj:`orngEnviron.addOnsDirUser`).
    """
    load_installed_addons_from_dir(Orange.misc.environ.add_ons_dir)
    load_installed_addons_from_dir(Orange.misc.environ.add_ons_dir_user)

def refresh_addons(reload_path=False):
    """
    Call add-on list change callbacks (ie. functions in
    :obj:`addon_refresh_callback`) and, optionally, refresh the python path
    (:obj:`sys.path`) with appropriate add-on directories (ie. call
    :obj:`addon_refresh_callback`).
    
    :param reload_path: determines whether python path should be refreshed.
    :type reload_path: boolean
    """
    if reload_path:
        add_addon_directories_to_path()
    for func in addon_refresh_callback:
        func()
        
# Registered add-ons support        
def __read_addons_list(addons_file, systemwide):
    if os.path.isfile(addons_file):
        name_path_list = [tuple([x.strip() for x in lne.split("\t")])
                          for lne in file(addons_file, "rt")]
        return [OrangeRegisteredAddOn(name, path, systemwide)
                for (name, path) in name_path_list]
    else:
        return []
    
def __read_addon_lists(userOnly=False):
    return __read_addons_list(os.path.join(Orange.misc.environ.orange_settings_dir, "add-ons.txt"),
                              False) + ([] if userOnly else
                                        __read_addons_list(os.path.join(Orange.misc.environ.install_dir, "add-ons.txt"),
                                                           True))

def __write_addon_lists(addons, user_only=False):
    file(os.path.join(Orange.misc.environ.orange_settings_dir, "add-ons.txt"), "wt").write("\n".join(["%s\t%s" % (a.name, a.directory) for a in addons if not a.systemwide]))
    if not user_only:
        file(os.path.join(Orange.misc.environ.install_dir        , "add-ons.txt"), "wt").write("\n".join(["%s\t%s" % (a.name, a.directory) for a in addons if     a.systemwide]))

def register_addon(name, path, add = True, refresh=True, systemwide=False):
    """
    Register the given path as an registered add-on with a given descriptive
    name. The operation is persistent, ie. on next :obj:`load_addons` call the
    path will still appear as registered.
    
    :param name: a descriptive name for the registered add-on.
    :type name: str
    
    :param path: path to be registered.
    :type path: str
    
    :param add: if False, the given path is UNREGISTERED instead of registered.
    :type add: boolean
    
    :param refresh: determines whether callbacks should be called after the
        procedure.
    :type refresh: boolean
    
    :param systemwide: determines whether the path is to be registered
        system-wide, i.e. for all users. Administrative privileges on the
        filesystem are usually needed for that.
    :type systemwide: boolean
    """
    if not add:
        unregister_addon(name, path, user_only=not systemwide)
    else:
        if os.path.isfile(path):
            path = os.path.dirname(path)
        __write_addon_lists([a for a in __read_addon_lists(user_only=not systemwide)
                             if a.name != name and a.directory != path] +\
                           ([OrangeRegisteredAddOn(name, path, systemwide)] or []),
                             user_only=not systemwide)
    
        global registered_addons
        registered_addons.append( OrangeRegisteredAddOn(name, path, systemwide) )
    if refresh:
        refresh_addons()

def unregister_addon(name, path, user_only=False):
    """
    Unregister the given path if it has been registered as an add-on with given
    descriptive name. The operation is persistent, ie. on next
    :obj:`load_addons` call the path will no longer appear as registered.
    
    :param name: a descriptive name of the registered add-on to be unregistered.
    :type name: str
    
    :param path: path to be unregistered.
    :type path: str

    :param user_only: determines whether the path to be unregistered is
        registered for this user only, ie. not system-wide. Administrative
        privileges on the filesystem are usually needed to unregister a
        system-wide registered add-on.
    :type systemwide: boolean
    """
    global registered_addons
    registered_addons = [ao for ao in registered_addons
                         if (ao.name!=name) or (ao.directory!=path) or
                         (user_only and ao.systemwide)]
    __write_addon_lists([a for a in __read_addon_lists(user_only=user_only)
                         if a.name != name and a.directory != path],
                         user_only=user_only)


def __get_registered_addons():
    return {'registered_addons': __read_addon_lists()}

load_addons()
globals().update(__get_registered_addons())

addon_refresh_callback = []
globals().update({'addon_refresh_callback': addon_refresh_callback})

add_addon_directories_to_path()

load_repositories(refresh=False)
