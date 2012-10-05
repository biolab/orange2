"""
Orange Canvas Configuration

"""

import os
import logging
import cPickle

log = logging.getLogger(__name__)

from PyQt4.QtGui import QDesktopServices
from PyQt4.QtCore import QCoreApplication, QSettings


def init():
    """Initialize the QCoreApplication.organizationDomain, applicationName,
    applicationVersion and the default settings format. Will only run once.

    """
    # Set application name, version and org. domain
    QCoreApplication.setOrganizationDomain("biolab.si")
    QCoreApplication.setApplicationName("Orange Canvas")
    QCoreApplication.setApplicationVersion("2.6")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    # Make it a null op.
    global init
    init = lambda: None

rc = {}

default_config = \
{
    "startup.show-splash-screen": True,
    "startup.show-welcome-dialog": True,
    "startup.style": None,
    "startup.stylesheet": "orange",
    "mainwindow.shop-properties-on-new-scheme": True,
    "mainwindow.use-native-theme": False,
    "mainwindow.qt-style": "default",
    "mainwindow.show-document-margins": False,
    "mainwindow.document-margins": 20,
    "mainwindow.document-shadow": True,
    "mainwindow.toolbox-dock-type": "toolbox",
    "mainwindow.toolbox-toolbar-position": "bottom",
    "mainwindow.toolbox-dock-floatable": False,
    "mainwindow.show-status-bar": False,
    "mainwindow.single-document-mode": True,
    "logging.level": "error",
}


def data_dir():
    """Return the application data directory.
    """
    init()
    datadir = QDesktopServices.storageLocation(QDesktopServices.DataLocation)
    datadir = unicode(datadir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    return datadir


def cache_dir():
    """Return the application cache directory.
    """
    init()
    datadir = QDesktopServices.storageLocation(QDesktopServices.DataLocation)
    datadir = unicode(datadir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    return datadir


def open_config():
    global rc
    app_dir = data_dir()
    filename = os.path.join(app_dir, "canvas-rc.pck")
    if os.path.exists(filename):
        with open(os.path.join(app_dir, "canvas-rc.pck"), "rb") as f:
            rc.update(cPickle.load(f))


def save_config():
    app_dir = data_dir()
    with open(os.path.join(app_dir, "canvas-rc.pck"), "wb") as f:
        cPickle.dump(rc, f)


def recent_schemes():
    """Return a list of recently accessed schemes.
    """
    app_dir = data_dir()
    recent_filename = os.path.join(app_dir, "recent.pck")
    recent = []
    if os.path.isdir(app_dir) and os.path.isfile(recent_filename):
        with open(recent_filename, "rb") as f:
            recent = cPickle.load(f)

    # Filter out files not found on the file system
    recent = [(title, path) for title, path in recent \
              if os.path.exists(path)]
    return recent


def save_recent_scheme_list(scheme_list):
    """Save the list of recently accessed schemes
    """
    app_dir = data_dir()
    recent_filename = os.path.join(app_dir, "recent.pck")

    if os.path.isdir(app_dir):
        with open(recent_filename, "wb") as f:
            cPickle.dump(scheme_list, f)
