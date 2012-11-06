"""
Orange Canvas main entry point

"""

import os
import sys
import re
import logging
import optparse
import cPickle
from contextlib import nested

import pkg_resources

from PyQt4.QtGui import QFont, QColor
from PyQt4.QtCore import QRect, QSettings, QDir

from Orange import OrangeCanvas
from Orange.OrangeCanvas.application.application import CanvasApplication
from Orange.OrangeCanvas.application.canvasmain import CanvasMainWindow

from Orange.OrangeCanvas.gui.splashscreen import SplashScreen, QPixmap
from Orange.OrangeCanvas.config import cache_dir
from Orange.OrangeCanvas import config
from Orange.OrangeCanvas.utils.redirect import redirect_stdout, redirect_stderr

from Orange.OrangeCanvas.registry import qt
from Orange.OrangeCanvas.registry import WidgetRegistry, set_global_registry
from Orange.OrangeCanvas.registry import cache

log = logging.getLogger(__name__)


def qt_logging_handle(msg_type, message):
    print msg_type, message


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    usage = "usage: %prog [options] [scheme_file]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--no-discovery",
                      action="store_true",
                      help="Don't run widget discovery "
                           "(use full cache instead)")

    parser.add_option("--force-discovery",
                      action="store_true",
                      help="Force full widget discovery "
                           "(invalidate cache)")
    parser.add_option("--no-welcome",
                      action="store_true",
                      help="Don't show welcome dialog.")
    parser.add_option("--no-splash",
                      action="store_true",
                      help="Don't show splash screen.")
    parser.add_option("-l", "--log-level",
                      help="Logging level (0, 1, 2, 3)",
                      type="int", default=1)
    parser.add_option("--no-redirect",
                      action="store_true",
                      help="Do not redirect stdout/err to canvas output view.")
    parser.add_option("--style",
                      help="QStyle to use",
                      type="str", default=None)
    parser.add_option("--stylesheet",
                      help="Application level CSS style sheet to use",
                      type="str", default="orange.qss")
    parser.add_option("--qt",
                      help="Additional arguments for QApplication",
                      type="str", default=None)

    (options, args) = parser.parse_args(argv)

    levels = [logging.CRITICAL,
              logging.WARN,
              logging.INFO,
              logging.DEBUG]

    logging.basicConfig(level=levels[options.log_level])

    log.info("Starting 'Orange Canvas' application.")

    qt_argv = ["orange-canvas"]

    if options.style is not None:
        qt_argv += ["-style", options.style]

    if options.qt is not None:
        qt_argv += options.qt.split()

    log.debug("Starting CanvasApplicaiton with argv = %r.", qt_argv)
    app = CanvasApplication(qt_argv)

    # Note: config.init must be called after the QApplication constructor
    config.init()
    settings = QSettings()

    stylesheet = options.stylesheet
    stylesheet_string = None

    if stylesheet != "none":
        if os.path.isfile(stylesheet):
            stylesheet_string = open(stylesheet, "rb").read()
        else:
            if not os.path.splitext(stylesheet)[1]:
                # no extension
                stylesheet = os.path.extsep.join([stylesheet, "qss"])

            pkg_name = OrangeCanvas.__name__
            resource = os.path.join("styles", stylesheet)

            if pkg_resources.resource_exists(pkg_name, resource):
                stylesheet_string = pkg_resources.resource_string(
                                        pkg_name, resource)
                base = pkg_resources.resource_filename(pkg_name, "styles")

                matches = re.findall(
                    r"^\s@([a-zA-Z0-9_]+?)=([a-zA-Z0-9_/]+?)$",
                    stylesheet_string,
                    re.MULTILINE)

                for prefix, search_path in matches:
                    QDir.addSearchPath(prefix, os.path.join(base, search_path))
                    log.info("Adding search path %r for prefix, %r",
                             search_path, prefix)
            else:
                log.info("%r style sheet not found.", stylesheet)

    if stylesheet_string is not None:
        app.setStyleSheet(stylesheet_string)

    # Add the default canvas_icons search path
    dirpath = os.path.abspath(os.path.dirname(OrangeCanvas.__file__))
    QDir.addSearchPath("canvas_icons", os.path.join(dirpath, "icons"))

    canvas_window = CanvasMainWindow()
    canvas_window.resize(1024, 650)

    if not options.force_discovery:
        reg_cache = cache.registry_cache()
    else:
        reg_cache = None

    widget_discovery = qt.QtWidgetDiscovery(cached_descriptions=reg_cache)

    widget_registry = qt.QtWidgetRegistry()

    widget_discovery.found_category.connect(
        widget_registry.register_category
    )
    widget_discovery.found_widget.connect(
        widget_registry.register_widget
    )

    want_splash = \
        settings.value("startup/show-splash-screen", True).toPyObject() and \
        not options.no_splash

    if want_splash:
        pm = QPixmap(pkg_resources.resource_filename(
                        __name__, "icons/orange-splash-screen.png")
                     )
        # Text rectangle in which to fit the message.
        rect = QRect(88, 193, 200, 20)
        splash_screen = SplashScreen(pixmap=pm, textRect=rect)
        splash_screen.setFont(QFont("Helvetica", 12))
        color = QColor("#FFD39F")

        def show_message(message):
            splash_screen.showMessage(message, color=color)

        widget_discovery.discovery_start.connect(splash_screen.show)
        widget_discovery.discovery_process.connect(show_message)
        widget_discovery.discovery_finished.connect(splash_screen.hide)

    log.info("Running widget discovery process.")

    cache_filename = os.path.join(cache_dir(), "widget-registry.pck")
    if options.no_discovery:
        widget_registry = cPickle.load(open(cache_filename, "rb"))
        widget_registry = qt.QtWidgetRegistry(widget_registry)
    else:
        widget_discovery.run()
        # Store cached descriptions
        cache.save_registry_cache(widget_discovery.cached_descriptions)
        cPickle.dump(WidgetRegistry(widget_registry),
                     open(cache_filename, "wb"))
    set_global_registry(widget_registry)
    canvas_window.set_widget_registry(widget_registry)
    canvas_window.show()

    want_welcome = \
        settings.value("welcomedialog/show-at-startup", True).toBool() \
        and not options.no_welcome

    canvas_window.raise_()

    if want_welcome and not args:
        canvas_window.welcome_dialog()

    elif args:
        log.info("Loading a scheme from the command line argument %r",
                 args[0])
        canvas_window.load_scheme(args[0])

    disable_redirect = \
        settings.value("mainwindow/no-stdout-redirect", False).toBool() or \
        options.no_redirect

    if not disable_redirect:
        output = canvas_window.output_view()
    else:
        output = None

    with nested(redirect_stdout(output), redirect_stderr(output)):
        log.info("Entering main event loop.")
        try:
            status = app.exec_()
        except BaseException:
            log.error("Error in main event loop.", exc_info=True)

    canvas_window.deleteLater()
    app.processEvents()
    app.flush()
    del canvas_window
    del app
    return status


if __name__ == "__main__":
    sys.exit(main())
