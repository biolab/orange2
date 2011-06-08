"""
Orange environment configuration
================================

"""

import os, sys
import ConfigParser

def _path_fix():
    """ Fix some common problems with $(PATH) and sys.path
    """
    if os.name == "nt":
        ## Move any miktex paths containing Qt's dll to the end of the %PATH%
        paths = os.environ["PATH"].split(";")
        paths.sort(lambda x,y: -1 if "PyQt4" in x else (1 if "miktex" in y and \
                        os.path.exists(os.path.join(y, "QtCore4.dll")) else 0))
        os.environ["PATH"] = ";".join(paths)
        
    if sys.platform == "darwin":
        ## PyQt4 installed from fink is installed in %{FINK_ROOT}lib/qt4-mac/lib/python${PYVERSION}/site-packages"
        posible_fink_pyqt4_path = os.path.join(sys.prefix, 
                "lib/qt4-mac/lib/python" + sys.version[:3] + "/site-packages")
        if os.path.exists(posible_fink_pyqt4_path):
            sys.path.append(posible_fink_pyqt4_path)
            
_path_fix()

def _get_default_env():
    """ Return a dictionary with default Orange environment."""
    version = "orange"
    version_display = "Orange 2.5"
    orange_no_deprecated_members = "False"
    
    try:
        install_dir = os.path.dirname(os.path.abspath(__file__)) # Orange/misc
        install_dir = os.path.dirname(install_dir) # Orange/
        install_dir = os.path.dirname(install_dir) # 
    except Exception: # Why should this happen??
        raise
        import orange
        install_dir = os.path.dirname(os.path.abspath(orange.__file__))

    doc_install_dir = os.path.join(install_dir, "doc") 

    canvas_install_dir = os.path.join(install_dir, "OrangeCanvas")
    widget_install_dir = os.path.join(install_dir, "OrangeWidgets")
    icons_install_dir = os.path.join(widget_install_dir, "icons")
    add_ons_dir = os.path.join(install_dir, "add-ons")

    home = os.path.expanduser("~/")
    
    if sys.platform == "win32":
        if home[-1] == ":":
            home += "\\"
        application_dir = os.environ["APPDATA"]
        output_dir = os.path.join(application_dir, version)
        default_reports_dir = os.path.join(home, "My Documents", "Orange Reports")
    elif sys.platform == "darwin":
        application_dir = os.path.join(home, "Library", "Application Support")
        output_dir = os.path.join(application_dir, version)
        default_reports_dir = os.path.join(home, "Library",
                                           "Application Support",
                                           version, "Reports")
    else:
        application_dir = home
        output_dir = os.path.join(home, "." + version)
        default_reports_dir = os.path.join(home, "orange-reports")

    add_ons_dir_user = os.path.join(output_dir, "add-ons")

    orange_settings_dir = output_dir
    canvas_settings_dir = os.path.join(output_dir, "OrangeCanvasQt4")
    widget_settings_dir = os.path.join(output_dir, "widgetSettingsQt4")
    
    if sys.platform == "darwin":
        buffer_dir = os.path.join(home, "Library", "Caches", version)
    else:
        buffer_dir = os.path.join(output_dir, "buffer")

    return locals()

_ALL_ENV_OPTIONS = ["version", "version_display", "is_canvas_installed",
                    "orange_no_deprecated_members"]

_ALL_DIR_OPTIONS = ["install_dir", "canvas_install_dir",
                    "widget_install_dir", "icons_install_dir",
                    "doc_install_dir", "add_ons_dir", "add_ons_dir_user",
                    "application_dir", "output_dir", "default_reports_dir",
                    "orange_settings_dir", "canvas_settings_dir",
                    "widget_settings_dir", "buffer_dir"]

def get_platform_option(section, option):
    """ Return the platform specific configuration `option` from `section`.
    """
    try:
        return parser.get(section + " " + sys.platform, option)
    except Exception:
        return parser.get(section, option)
    
def _configure_env(defaults=None):
    """ Apply the configuration files on the default environment
    and return the instance of SafeConfigParser
     
    """
    if defaults is None:
        defaults = dict(os.environ)
        defaults.update(_get_default_env())
        
    parser = ConfigParser.SafeConfigParser(defaults)
    global_cfg = os.path.join(defaults["install_dir"], "orangerc.cfg")
    if not parser.read([global_cfg]):
        pass
#        import warnings
#        warnings.warn("Could not read global orange configuration file.")

    # In case the global_cfg does not exist or is empty
    if not parser.has_section("directories"):
        parser.add_section("directories")
    if not parser.has_section("Orange"):
        parser.add_section("Orange")
        
    platform = sys.platform
    try:
        application_dir = parser.get("directories " + platform, "application_dir")
    except Exception:
        application_dir = parser.get("directories", "application_dir")
    
    parser.read([os.path.join(application_dir, "orangerc.cfg"),
                 os.path.expanduser("~/.orangerc.cfg")])
    
    return parser

parser = _configure_env()

for dirname in _ALL_DIR_OPTIONS:
    globals()[dirname] = get_platform_option("directories", dirname)
    

if not os.path.isdir(widget_install_dir) or not os.path.isdir(widget_install_dir):
    # Canvas and widgets are not installed
    canvas_install_dir = None
    widget_install_dir = None
    canvas_settings_dir = None
    widget_settings_dir = None
    is_canvas_installed = False
else:
    is_canvas_installed = True
    
if not os.path.isdir(icons_install_dir):
    icons_install_dir = ""
    
version = parser.get("Orange", "version")
version_display = parser.get("Orange", "version_display")
orange_no_deprecated_members = parser.getboolean("Orange", "orange_no_deprecated_members")
version = get_platform_option("Orange", "version")
    
# Create the directories if missing
_directories_to_create = ["application_dir", "orange_settings_dir",
        "buffer_dir", "widget_settings_dir", "canvas_settings_dir",
        "default_reports_dir"]

for dname in _directories_to_create:
    dname = globals()[dname]
    if dname is not None and not os.path.isdir(dname):
        try:
            os.makedirs(dname)        # Vista has roaming profiles that will say that this folder does not exist and will then fail to create it, because it exists...
        except Exception:
            pass
        
def samepath(path1, path2):
    """ Do the paths path1 and path2 point to the same location.
    """
    return os.path.normcase(os.path.realpath(os.path.normpath(path1))) == \
           os.path.normcase(os.path.realpath(os.path.normpath(path2)))

def add_orange_directories_to_path():
    """Add orange directory paths to sys.path."""
    paths_to_add = [install_dir]

    if canvas_install_dir is not None:
        paths_to_add.append(canvas_install_dir)

    # Instead of doing this OrangeWidgets should be a package
    if widget_install_dir is not None and os.path.isdir(widget_install_dir):
        paths_to_add.append(widget_install_dir)
        default_widgets_dirs = [os.path.join(widget_install_dir, x) \
                                for x in os.listdir(widget_install_dir) \
                                if os.path.isdir(os.path.join(widget_install_dir, x))]
        paths_to_add.extend(default_widgets_dirs)

    for path in paths_to_add:
        if os.path.isdir(path) and not any([samepath(path, x) for x in sys.path]):
            sys.path.insert(0, path)
            
add_orange_directories_to_path()
directories = dict([(dname, globals()[dname]) for dname in _ALL_DIR_OPTIONS])
