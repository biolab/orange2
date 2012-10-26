"""
Orange Canvas Main Window

"""
import os
import sys
import logging
import traceback
import operator

import pkg_resources

from PyQt4.QtGui import (
    QMainWindow, QWidget, QAction, QActionGroup, QMenu, QMenuBar, QDialog,
    QFileDialog, QMessageBox, QVBoxLayout, QSizePolicy, QColor, QKeySequence,
    QIcon, QToolBar, QDockWidget, QDesktopServices, QUndoGroup
)

from PyQt4.QtCore import (
    Qt, QEvent, QSize, QUrl, QSettings
)

from PyQt4.QtCore import pyqtProperty as Property


from ..gui.dropshadow import DropShadowFrame
from ..gui.dock import CollapsibleDockWidget

from .canvastooldock import CanvasToolDock, QuickCategoryToolbar
from .aboutdialog import AboutDialog
from .schemeinfo import SchemeInfoDialog
from ..document.schemeedit import SchemeEditWidget

from ..scheme import widgetsscheme

from . import welcomedialog
from ..preview import previewdialog, previewmodel

from .. import config

log = logging.getLogger(__name__)

# TODO: Orange Version in the base link

BASE_LINK = "http://orange.biolab.si/"

LINKS = \
    {"start-using": BASE_LINK + "start-using/",
     "tutorial": BASE_LINK + "tutorial/",
     "reference": BASE_LINK + "doc/"
     }


def style_icons(widget, standard_pixmap):
    """Return the Qt standard pixmap icon.
    """
    return QIcon(widget.style().standardPixmap(standard_pixmap))


def canvas_icons(name):
    """Return the named canvas icon.
    """
    return QIcon(pkg_resources.resource_filename(
                  config.__name__,
                  os.path.join("icons", name))
                 )


def message_critical(text, title=None, informative_text=None, details=None,
                     buttons=None, default_button=None, exc_info=False,
                     parent=None):
    """Show a critical message.
    """
    if not text:
        text = "An unexpected error occurred."

    if title is None:
        title = "Error"

    return message(QMessageBox.Critical, text, title, informative_text,
                   details, buttons, default_button, exc_info, parent)


def message_warning(text, title=None, informative_text=None, details=None,
                    buttons=None, default_button=None, exc_info=False,
                    parent=None):
    """Show a warning message.
    """
    if not text:
        import random
        text_candidates = ["Death could come at any moment.",
                           "Murphy lurks about. Remember to save frequently."
                           ]
        text = random.choice(text_candidates)

    if title is not None:
        title = "Warning"

    return message(QMessageBox.Warning, text, title, informative_text,
                   details, buttons, default_button, exc_info, parent)


def message_information(text, title=None, informative_text=None, details=None,
                        buttons=None, default_button=None, exc_info=False,
                        parent=None):
    """Show an information message box.
    """
    if title is None:
        title = "Information"
    if not text:
        text = "I am not a number."

    return message(QMessageBox.Information, text, title, informative_text,
                   details, buttons, default_button, exc_info, parent)


def message_question(text, title, informative_text=None, details=None,
                     buttons=None, default_button=None, exc_info=False,
                     parent=None):
    """Show an message box asking the user to select some
    predefined course of action (set by buttons argument).

    """
    return message(QMessageBox.Question, text, title, informative_text,
                   details, buttons, default_button, exc_info, parent)


def message(icon, text, title=None, informative_text=None, details=None,
            buttons=None, default_button=None, exc_info=False, parent=None):
    """Show a message helper function.
    """
    if title is None:
        title = "Message"
    if not text:
        text = "I am neither a postman nor a doctor."

    if buttons is None:
        buttons = QMessageBox.Ok

    if details is None and exc_info:
        details = traceback.format_exc(limit=20)

    mbox = QMessageBox(icon, title, text, buttons, parent)

    if informative_text:
        mbox.setInformativeText(informative_text)

    if details:
        mbox.setDetailedText(details)

    if default_button is not None:
        mbox.setDefaultButton(default_button)

    return mbox.exec_()


class FakeToolBar(QToolBar):
    """A Toolbar with no contents (used to reserve top and bottom margins
    on the main window).

    """
    def __init__(self, *args, **kwargs):
        QToolBar.__init__(self, *args, **kwargs)
        self.setFloatable(False)
        self.setMovable(False)

        # Don't show the tool bar action in the main window's
        # context menu.
        self.toggleViewAction().setVisible(False)

    def paintEvent(self, event):
        # Do nothing.
        pass


class CanvasMainWindow(QMainWindow):
    SETTINGS_VERSION = 1

    def __init__(self, *args):
        QMainWindow.__init__(self, *args)

        self.__scheme_margins_enabled = True

        self.widget_registry = None
        self.last_scheme_dir = None

        self.recent_schemes = config.recent_schemes()

        self.setup_ui()

        self.resize(800, 600)

    def setup_ui(self):
        """Setup main canvas ui
        """
        QSettings.setDefaultFormat(QSettings.IniFormat)
        settings = QSettings()
        settings.beginGroup("canvasmainwindow")

        log.info("Setting up Canvas main window.")

        self.setup_actions()
        self.setup_menu()

        # Two dummy tool bars to reserve space
        self.__dummy_top_toolbar = FakeToolBar(
                            objectName="__dummy_top_toolbar")
        self.__dummy_bottom_toolbar = FakeToolBar(
                            objectName="__dummy_bottom_toolbar")

        self.__dummy_top_toolbar.setFixedHeight(20)
        self.__dummy_bottom_toolbar.setFixedHeight(20)

        self.addToolBar(Qt.TopToolBarArea, self.__dummy_top_toolbar)
        self.addToolBar(Qt.BottomToolBarArea, self.__dummy_bottom_toolbar)

        # Create an empty initial scheme inside a container with fixed
        # margins.
        w = QWidget()
        w.setLayout(QVBoxLayout())
        w.layout().setContentsMargins(20, 0, 10, 0)

        self.scheme_widget = SchemeEditWidget()
        self.scheme_widget.setScheme(widgetsscheme.WidgetsScheme())

        self.undo_group.addStack(self.scheme_widget.undoStack())
        self.undo_group.setActiveStack(self.scheme_widget.undoStack())

        w.layout().addWidget(self.scheme_widget)

        self.setCentralWidget(w)

        # Drop shadow around the scheme document
        frame = DropShadowFrame(radius=15)
        frame.setColor(QColor(0, 0, 0, 100))
        frame.setWidget(self.scheme_widget)

        # Main window title and title icon.
        self.setWindowTitle(self.scheme_widget.scheme().title)
        self.scheme_widget.titleChanged.connect(self.setWindowTitle)

        self.setWindowIcon(canvas_icons("Get Started.svg"))

        # QMainWindow's Dock widget
        self.dock_widget = CollapsibleDockWidget(objectName="main-area-dock")
        self.dock_widget.setFeatures(QDockWidget.DockWidgetMovable | \
                                     QDockWidget.DockWidgetClosable)

        # Main canvas tool dock (with widget toolbox, common actions.
        # This is the widget that is shown when the dock is expanded.
        canvas_tool_dock = CanvasToolDock(objectName="canvas-tool-dock")
        canvas_tool_dock.setSizePolicy(QSizePolicy.Fixed,
                                       QSizePolicy.MinimumExpanding)

        # Bottom tool bar
        self.canvas_toolbar = canvas_tool_dock.toolbar
        self.canvas_toolbar.setIconSize(QSize(25, 25))
        self.canvas_toolbar.setFixedHeight(28)
        self.canvas_toolbar.layout().setSpacing(1)

        # Widgets tool box
        self.widgets_tool_box = canvas_tool_dock.toolbox
        self.widgets_tool_box.setObjectName("canvas-toolbox")
        self.widgets_tool_box.setTabButtonHeight(30)
        self.widgets_tool_box.setTabIconSize(QSize(26, 26))
        self.widgets_tool_box.setButtonSize(QSize(64, 84))
        self.widgets_tool_box.setIconSize(QSize(48, 48))

        self.widgets_tool_box.triggered.connect(
            self.on_tool_box_widget_activated
        )

        self.widgets_tool_box.hovered.connect(
            self.on_tool_box_widget_hovered
        )

        self.dock_help = canvas_tool_dock.help
        self.dock_help.setMaximumHeight(150)
        self.dock_help.document().setDefaultStyleSheet("h3 {color: orange;}")

        self.dock_help_action = canvas_tool_dock.toogleQuickHelpAction()
        self.dock_help_action.setText(self.tr("Show Help"))
        self.dock_help_action.setIcon(canvas_icons("Info.svg"))

        self.canvas_tool_dock = canvas_tool_dock

        # Dock contents when collapsed (a quick category tool bar, ...)
        dock2 = QWidget(objectName="canvas-quick-dock")
        dock2.setLayout(QVBoxLayout())
        dock2.layout().setContentsMargins(0, 0, 0, 0)
        dock2.layout().setSpacing(0)
        dock2.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

        self.quick_category = QuickCategoryToolbar()
        self.quick_category.setButtonSize(QSize(38, 30))
        self.quick_category.actionTriggered.connect(
            self.on_quick_category_action
        )

        dock_actions = [self.show_properties_action,
                        self.canvas_zoom_action,
                        self.canvas_align_to_grid_action,
                        self.canvas_arrow_action,
                        self.canvas_text_action,
                        self.freeze_action,
                        self.dock_help_action]

        # Tool bar in the collapsed dock state (has the same actions as
        # the tool bar in the CanvasToolDock
        actions_toolbar = QToolBar(orientation=Qt.Vertical)
        actions_toolbar.setFixedWidth(38)
        actions_toolbar.layout().setSpacing(0)

        actions_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)

        for action in dock_actions:
            actions_toolbar.addAction(action)
            self.canvas_toolbar.addAction(action)
            button = actions_toolbar.widgetForAction(action)
            button.setFixedSize(38, 30)

        dock2.layout().addWidget(self.quick_category)
        dock2.layout().addWidget(actions_toolbar)

        self.dock_widget.setAnimationEnabled(False)
        self.dock_widget.setExpandedWidget(self.canvas_tool_dock)
        self.dock_widget.setCollapsedWidget(dock2)
        self.dock_widget.setExpanded(True)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
        self.dock_widget.dockLocationChanged.connect(
            self._on_dock_location_changed
        )

        self.setMinimumSize(600, 500)

        state = settings.value("state")
        if state.isValid():
            self.restoreState(state.toByteArray(),
                              version=self.SETTINGS_VERSION)

        self.dock_widget.setExpanded(
            settings.value("canvasdock/expanded", True).toBool()
        )

        self.toogle_margins_action.setChecked(
            settings.value("scheme_margins_enabled", True).toBool()
        )

        self.last_scheme_dir = \
            settings.value("last_scheme_dir", None).toPyObject()

        if self.last_scheme_dir is not None and \
                not os.path.exists(self.last_scheme_dir):
            # if directory no longer exists reset the saved location.
            self.last_scheme_dir = None

    def setup_actions(self):
        """Initialize main window actions.
        """

        self.new_action = \
            QAction(self.tr("New"), self,
                    objectName="action-new",
                    toolTip=self.tr("Open a new scheme."),
                    triggered=self.new_scheme,
                    shortcut=QKeySequence.New,
                    icon=canvas_icons("New.svg")
                    )

        self.open_action = \
            QAction(self.tr("Open"), self,
                    objectName="action-open",
                    toolTip=self.tr("Open a scheme."),
                    triggered=self.open_scheme,
                    shortcut=QKeySequence.Open,
                    icon=canvas_icons("Open.svg")
                    )

        self.save_action = \
            QAction(self.tr("Save"), self,
                    objectName="action-save",
                    toolTip=self.tr("Save current scheme."),
                    triggered=self.save_scheme,
                    shortcut=QKeySequence.Save,
                    )

        self.save_as_action = \
            QAction(self.tr("Save As ..."), self,
                    objectName="action-save-as",
                    toolTip=self.tr("Save current scheme as."),
                    triggered=self.save_scheme_as,
                    shortcut=QKeySequence.SaveAs,
                    )

        self.quit_action = \
            QAction(self.tr("Quit"), self,
                    objectName="quit-action",
                    toolTip=self.tr("Quit Orange Canvas."),
                    triggered=self.quit,
                    menuRole=QAction.QuitRole,
                    shortcut=QKeySequence.Quit,
                    )

        self.welcome_action = \
            QAction(self.tr("Welcome"), self,
                    objectName="welcome-action",
                    toolTip=self.tr("Show welcome screen."),
                    triggered=self.welcome_dialog,
                    )

        self.get_started_action = \
            QAction(self.tr("Get Started"), self,
                    objectName="get-started-action",
                    toolTip=self.tr("View a 'Getting Started' video."),
                    triggered=self.get_started,
                    icon=canvas_icons("Get Started.svg")
                    )

        self.tutorials_action = \
            QAction(self.tr("Tutorial"), self,
                    objectName="tutorial-action",
                    toolTip=self.tr("View tutorial."),
                    triggered=self.tutorial,
                    icon=canvas_icons("Tutorials.svg")
                    )

        self.documentation_action = \
            QAction(self.tr("Documentation"), self,
                    objectName="documentation-action",
                    toolTip=self.tr("View reference documentation."),
                    triggered=self.documentation,
                    icon=canvas_icons("Documentation.svg")
                    )

        self.about_action = \
            QAction(self.tr("About"), self,
                    objectName="about-action",
                    toolTip=self.tr("Show about dialog."),
                    triggered=self.open_about,
                    menuRole=QAction.AboutRole,
                    )

        # Action group for for recent scheme actions
        self.recent_scheme_action_group = \
            QActionGroup(self, exclusive=False,
                         objectName="recent-action-group",
                         triggered=self._on_recent_scheme_action)

        self.recent_action = \
            QAction(self.tr("Browse Recent"), self,
                    objectName="recent-action",
                    toolTip=self.tr("Browse and open a recent scheme."),
                    triggered=self.recent_scheme,
                    shortcut=QKeySequence(Qt.ControlModifier | \
                                          (Qt.ShiftModifier | Qt.Key_R)),
                    icon=canvas_icons("Recent.svg")
                    )

        self.reload_last_action = \
            QAction(self.tr("Reload Last Scheme"), self,
                    objectName="reload-last-action",
                    toolTip=self.tr("Reload last open scheme."),
                    triggered=self.reload_last,
                    shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_R)
                    )

        self.clear_recent_action = \
            QAction(self.tr("Clear Menu"), self,
                    objectName="clear-recent-menu-action",
                    toolTip=self.tr("Clear recent menu."),
                    triggered=self.clear_recent_schemes
                    )

        self.show_properties_action = \
            QAction(self.tr("Show Properties"), self,
                    objectName="show-properties-action",
                    toolTip=self.tr("Show scheme properties."),
                    triggered=self.show_scheme_properties,
                    shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_I),
                    icon=canvas_icons("Document Info.svg")
                    )

        self.canvas_settings_action = \
            QAction(self.tr("Settings"), self,
                    objectName="canvas-settings-action",
                    toolTip=self.tr("Set application settings."),
                    triggered=self.open_canvas_settings,
                    menuRole=QAction.PreferencesRole,
                    shortcut=QKeySequence.Preferences
                    )

        self.undo_group = QUndoGroup(self)
        self.undo_action = self.undo_group.createUndoAction(self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.redo_action = self.undo_group.createRedoAction(self)
        self.redo_action.setShortcut(QKeySequence.Redo)

        self.select_all_action = \
            QAction(self.tr("Select All"), self,
                    objectName="select-all-action",
                    triggered=self.select_all,
                    shortcut=QKeySequence.SelectAll,
                    )

        self.open_widget_action = \
            QAction(self.tr("Open"), self,
                    objectName="open-widget-action",
                    triggered=self.open_widget,
                    )

        self.rename_widget_action = \
            QAction(self.tr("Rename"), self,
                    objectName="rename-widget-action",
                    triggered=self.rename_widget,
                    toolTip="Rename a widget",
                    shortcut=QKeySequence(Qt.Key_F2)
                    )

        self.remove_widget_action = \
            QAction(self.tr("Remove"), self,
                    objectName="remove-action",
                    triggered=self.remove_selected,
                    )

        delete_shortcuts = [Qt.Key_Delete,
                            Qt.ControlModifier + Qt.Key_Backspace]

        if sys.platform == "darwin":
            # Command Backspace should be the first
            # (visible shortcut in the menu)
            delete_shortcuts.reverse()

        self.remove_widget_action.setShortcuts(delete_shortcuts)

        self.widget_help_action = \
            QAction(self.tr("Help"), self,
                    objectName="widget-help-action",
                    triggered=self.widget_help,
                    toolTip=self.tr("Show widget help."),
                    shortcut=QKeySequence.HelpContents,
                    )

        if sys.platform == "darwin":
            # Actions for native Mac OSX look and feel.
            self.minimize_action = \
                QAction(self.tr("Minimize"), self,
                        triggered=self.showMinimized,
                        shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_M)
                        )

            self.zoom_action = \
                QAction(self.tr("Zoom"), self,
                        objectName="application-zoom",
                        triggered=self.toggleMaximized,
                        )

        # Canvas Dock actions
        self.canvas_zoom_action = \
            QAction(self.tr("Zoom"), self,
                    objectName="canvas-zoom-actions",
                    checkable=True,
                    shortcut=QKeySequence.ZoomIn,
                    toolTip=self.tr("Zoom in the scheme."),
                    triggered=self.set_canvas_view_zoom,
                    icon=canvas_icons("Search.svg")
                    )

        self.canvas_align_to_grid_action = \
            QAction(self.tr("Clean Up"), self,
                    objectName="canvas-align-to-grid-action",
                    toolTip=self.tr("Align widget to a grid."),
                    triggered=self.align_to_grid,
                    icon=canvas_icons("Grid.svg")
                    )

        self.canvas_arrow_action = \
            QAction(self.tr("Arrow"), self,
                    objectName="canvas-arrow-action",
                    toolTip=self.tr("Add an arrow annotation to the scheme."),
                    triggered=self.new_arrow_annotation,
                    icon=canvas_icons("Arrow.svg")
                    )

        self.canvas_text_action = \
            QAction(self.tr("Text"), self,
                    objectName="canvas-text-action",
                    toolTip=self.tr("Add a text annotation to the scheme."),
                    triggered=self.new_text_annotation,
                    icon=canvas_icons("Text Size.svg")
                    )

        self.freeze_action = \
            QAction(self.tr("Freeze"), self,
                    objectName="signal-freeze-action",
                    checkable=True,
                    toolTip=self.tr("Freeze signal propagation."),
                    triggered=self.set_signal_freeze,
                    icon=canvas_icons("Pause.svg")
                    )

        # Gets assigned in setup_ui (the action is defined in CanvasToolDock)
        # TODO: This is bad (should be moved here).
        self.dock_help_action = None

        self.toogle_margins_action = \
            QAction(self.tr("Show Scheme Margins"), self,
                    checkable=True,
                    checked=True,
                    toolTip=self.tr("Show margins around the scheme view."),
                    toggled=self.set_scheme_margins_enabled
                    )

    def setup_menu(self):
        menu_bar = QMenuBar()

        # File menu
        file_menu = QMenu(self.tr("&File"), menu_bar)
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.reload_last_action)

        # File -> Open Recent submenu
        self.recent_menu = QMenu(self.tr("Open Recent"), file_menu)
        file_menu.addMenu(self.recent_menu)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.show_properties_action)
        file_menu.addAction(self.quit_action)

        self.recent_menu.addAction(self.recent_action)

        # Store the reference to separator for inserting recent
        # schemes into the menu in `add_recent_scheme`.
        self.recent_menu_begin = self.recent_menu.addSeparator()

        # Add recent items.
        for title, filename in self.recent_schemes:
            action = QAction(title, self, toolTip=filename)
            action.setData(filename)
            self.recent_menu.addAction(action)
            self.recent_scheme_action_group.addAction(action)

        self.recent_menu.addSeparator()
        self.recent_menu.addAction(self.clear_recent_action)
        menu_bar.addMenu(file_menu)

        # Edit menu
        self.edit_menu = QMenu("&Edit", menu_bar)
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.select_all_action)
        menu_bar.addMenu(self.edit_menu)

        # View menu
        self.view_menu = QMenu(self.tr("&View"), self)
        self.toolbox_menu = QMenu(self.tr("Widget Toolbox Style"),
                                  self.view_menu)
        self.toolbox_menu_group = \
            QActionGroup(self, objectName="toolbox-menu-group")

        a1 = self.toolbox_menu.addAction(self.tr("Tool Box"))
        a2 = self.toolbox_menu.addAction(self.tr("Tool List"))
        self.toolbox_menu_group.addAction(a1)
        self.toolbox_menu_group.addAction(a2)

        self.view_menu.addMenu(self.toolbox_menu)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.toogle_margins_action)
        menu_bar.addMenu(self.view_menu)

        # Options menu
        self.options_menu = QMenu(self.tr("&Options"), self)
        self.options_menu.addAction(self.tr("Show Output"))
#        self.options_menu.addAction("Add-ons")
#        self.options_menu.addAction("Developers")
#        self.options_menu.addAction("Run Discovery")
#        self.options_menu.addAction("Show Canvas Log")
#        self.options_menu.addAction("Attach Python Console")
        self.options_menu.addSeparator()
        self.options_menu.addAction(self.canvas_settings_action)

        # Widget menu
        self.widget_menu = QMenu(self.tr("Widget"), self)
        self.widget_menu.addAction(self.open_widget_action)
        self.widget_menu.addSeparator()
        self.widget_menu.addAction(self.rename_widget_action)
        self.widget_menu.addAction(self.remove_widget_action)
        self.widget_menu.addSeparator()
        self.widget_menu.addAction(self.widget_help_action)
        menu_bar.addMenu(self.widget_menu)

        if sys.platform == "darwin":
            # Mac OS X native look and feel.
            self.window_menu = QMenu(self.tr("Window"), self)
            self.window_menu.addAction(self.minimize_action)
            self.window_menu.addAction(self.zoom_action)
            menu_bar.addMenu(self.window_menu)

        menu_bar.addMenu(self.options_menu)

        # Help menu.
        self.help_menu = QMenu(self.tr("&Help"), self)
        self.help_menu.addAction(self.about_action)
        self.help_menu.addAction(self.welcome_action)
        self.help_menu.addAction(self.tutorials_action)
        self.help_menu.addAction(self.documentation_action)
        menu_bar.addMenu(self.help_menu)

        self.setMenuBar(menu_bar)

    def set_widget_registry(self, widget_registry):
        """Set widget registry.
        """
        if self.widget_registry is not None:
            # Clear the dock widget and popup.
            pass

        self.widget_registry = widget_registry
        self.widgets_tool_box.setModel(widget_registry.model())
        self.quick_category.setModel(widget_registry.model())

        self.scheme_widget.setRegistry(widget_registry)

    def set_quick_help_text(self, text):
        self.canvas_tool_dock.help.setText(text)

    def current_document(self):
        return self.scheme_widget

    def on_tool_box_widget_activated(self, action):
        """A widget action in the widget toolbox has been activated.
        """
        widget_desc = action.data().toPyObject()
        if widget_desc:
            scheme_widget = self.current_document()
            if scheme_widget:
                scheme_widget.createNewNode(widget_desc)

    def on_tool_box_widget_hovered(self, action):
        """Mouse is over a widget in the widget toolbox
        """
        widget_desc = action.data().toPyObject()
        title = ""
        help_text = ""
        if widget_desc:
            title = widget_desc.name
            description = widget_desc.help
            if not help_text:
                description = widget_desc.description

            template = "<h3>{title}</h3>" + \
                       "<p>{description}</p>" + \
                       "<a href=''>more...</a>"
            help_text = template.format(title=title, description=description)
            # TODO: 'More...' link
        self.set_quick_help_text(help_text)

    def on_quick_category_action(self, action):
        """The quick category menu action triggered.
        """
        category = action.text()
        for i in range(self.widgets_tool_box.count()):
            cat_act = self.widgets_tool_box.tabAction(i)
            if cat_act.text() == category:
                if not cat_act.isChecked():
                    # Trigger the action to expand the tool grid contained
                    # within.
                    cat_act.trigger()

            else:
                if cat_act.isChecked():
                    # Trigger the action to hide the tool grid contained
                    # within.
                    cat_act.trigger()

        self.dock_widget.expand()

    def set_scheme_margins_enabled(self, enabled):
        """Enable/disable the margins around the scheme document.
        """
        if self.__scheme_margins_enabled != enabled:
            self.__scheme_margins_enabled = enabled
            self.__update_scheme_margins()

    def scheme_margins_enabled(self):
        return self.__scheme_margins_enabled

    scheme_margins_enabled = Property(bool,
                                      fget=scheme_margins_enabled,
                                      fset=set_scheme_margins_enabled)

    def __update_scheme_margins(self):
        """Update the margins around the scheme document.
        """
        enabled = self.__scheme_margins_enabled
        self.__dummy_top_toolbar.setVisible(enabled)
        self.__dummy_bottom_toolbar.setVisible(enabled)
        central = self.centralWidget()

        margin = 20 if enabled else 0

        if self.dockWidgetArea(self.dock_widget) == Qt.LeftDockWidgetArea:
            margins = (margin / 2, 0, margin, 0)
        else:
            margins = (margin, 0, margin / 2, 0)

        central.layout().setContentsMargins(*margins)

    #################
    # Action handlers
    #################
    def new_scheme(self):
        """New scheme. Return QDialog.Rejected if the user canceled
        the operation and QDialog.Accepted otherwise.

        """
        document = self.current_document()
        if document.isModified():
            # Ask for save changes
            if self.ask_save_changes() == QDialog.Rejected:
                return QDialog.Rejected

        new_scheme = widgetsscheme.WidgetsScheme()
        scheme_doc_widget = self.current_document()
        scheme_doc_widget.setScheme(new_scheme)

        if config.rc.get("mainwindow.show-properties-on-new-scheme", True):
            self.show_properties_action.trigger()

        return QDialog.Accepted

    def open_scheme(self):
        """Open a new scheme. Return QDialog.Rejected if the user canceled
        the operation and QDialog.Accepted otherwise.

        """
        document = self.current_document()
        if document.isModified():
            if self.ask_save_changes() == QDialog.Rejected:
                return QDialog.Rejected

        if self.last_scheme_dir is None:
            # Get user 'Documents' folder
            start_dir = QDesktopServices.storageLocation(
                            QDesktopServices.DocumentsLocation)
        else:
            start_dir = self.last_scheme_dir

        # TODO: Use a dialog instance and use 'addSidebarUrls' to
        # set one or more extra sidebar locations where Schemes are stored.
        # Also use setHistory
        filename = QFileDialog.getOpenFileName(
            self, self.tr("Open Orange Scheme File"),
            start_dir, self.tr("Orange Scheme (*.ows)"),
        )

        if filename:
            self.load_scheme(filename)
            return QDialog.Accepted
        else:
            return QDialog.Rejected

    def load_scheme(self, filename):
        """Load a scheme from a file (`filename`) into the current
        document.

        """
        filename = unicode(filename)
        dirname = os.path.dirname(filename)

        self.last_scheme_dir = dirname

        new_scheme = widgetsscheme.WidgetsScheme()
        try:
            new_scheme.load_from(open(filename, "rb"))
            new_scheme.path = filename
        except Exception:
            message_critical(
                 self.tr("Could not load Orange Scheme file"),
                 title=self.tr("Error"),
                 informative_text=self.tr("An unexpected error occurred"),
                 exc_info=True,
                 parent=self)
            return

        scheme_doc_widget = self.current_document()
        scheme_doc_widget.setScheme(new_scheme)

        self.add_recent_scheme(new_scheme)

    def reload_last(self):
        """Reload last opened scheme. Return QDialog.Rejected if the
        user canceled the operation and QDialog.Accepted otherwise.

        """
        document = self.current_document()
        if document.isModified():
            if self.ask_save_changes() == QDialog.Rejected:
                return QDialog.Rejected

        # TODO: Search for a temp backup scheme with per process
        # locking.
        if self.recent_schemes:
            self.load_scheme(self.recent_schemes[0][1])

        return QDialog.Accepted

    def ask_save_changes(self):
        """Ask the user to save the changes to the current scheme.
        Return QDialog.Accepted if the scheme was successfully saved
        or the user selected to discard the changes. Otherwise return
        QDialog.Rejected.

        """
        document = self.current_document()

        selected = message_question(
            self.tr("Do you want to save the changes you made to scheme %r?") \
                    % document.scheme().title,
            self.tr("Save Changes?"),
            self.tr("If you do not save your changes will be lost"),
            buttons=QMessageBox.Save | QMessageBox.Cancel | \
                    QMessageBox.Discard,
            default_button=QMessageBox.Save,
            parent=self)

        if selected == QMessageBox.Save:
            return self.save_scheme()
        elif selected == QMessageBox.Discard:
            return QDialog.Accepted
        elif selected == QMessageBox.Cancel:
            return QDialog.Rejected

    def save_scheme(self):
        """Save the current scheme. If the scheme does not have an associated
        path then prompt the user to select a scheme file. Return
        QDialog.Accepted if the scheme was successfully saved and
        QDialog.Rejected if the user canceled the file selection.

        """
        document = self.current_document()
        curr_scheme = document.scheme()

        if curr_scheme.path:
            curr_scheme.save_to(open(curr_scheme.path, "wb"))
            document.setModified(False)
            return QDialog.Accepted
        else:
            return self.save_scheme_as()

    def save_scheme_as(self):
        """Save the current scheme by asking the user for a filename.
        Return QFileDialog.Accepted if the scheme was saved successfully
        and QFileDialog.Rejected if not.

        """
        curr_scheme = self.current_document().scheme()

        if curr_scheme.path:
            start_dir = curr_scheme.path
        elif self.last_scheme_dir is not None:
            start_dir = self.last_scheme_dir
        else:
            start_dir = QDesktopServices.storageLocation(
                            QDesktopServices.DocumentsLocation)

        filename = QFileDialog.getSaveFileName(
            self, self.tr("Save Orange Scheme File"),
            start_dir, self.tr("Orange Scheme (*.ows)")
        )

        if filename:
            filename = unicode(filename)
            dirname, basename = os.path.split(filename)
            self.last_scheme_dir = dirname

            try:
                curr_scheme.save_to(open(filename, "wb"))
            except Exception:
                log.error("Error saving %r to %r", curr_scheme, filename,
                          exc_info=True)
                # Also show a message box
                # TODO: should handle permission errors with a
                # specialized messages.
                message_critical(
                     self.tr("An error occurred while trying to save the %r "
                             "scheme to %r" % \
                             (curr_scheme.title, basename)),
                     title=self.tr("Error saving %r") % basename,
                     exc_info=True,
                     parent=self)
                return QFileDialog.Rejected

            curr_scheme.path = filename
            if not curr_scheme.title:
                curr_scheme.title = os.path.splitext(basename)[0]

            self.add_recent_scheme(curr_scheme)
            return QFileDialog.Accepted
        else:
            return QFileDialog.Rejected

    def get_started(self, *args):
        """Show getting started video
        """
        url = QUrl(LINKS["start-using"])
        QDesktopServices.openUrl(url)

    def tutorial(self, *args):
        """Show tutorial.
        """
        url = QUrl(LINKS["tutorial"])
        QDesktopServices.openUrl(url)

    def documentation(self, *args):
        """Show reference documentation.
        """
        url = QUrl(LINKS["tutorial"])
        QDesktopServices.openUrl(url)

    def recent_scheme(self, *args):
        """Browse recent schemes. Return QDialog.Rejected if the user
        canceled the operation and QDialog.Accepted otherwise.

        """
        items = [previewmodel.PreviewItem(name=title, path=path)
                 for title, path in self.recent_schemes]
        model = previewmodel.PreviewModel(items=items)

        dialog = previewdialog.PreviewDialog(self)
        dialog.setWindowTitle(self.tr("Recent Schemes"))
        dialog.setModel(model)

        model.delayedScanUpdate()

        status = dialog.exec_()

        if status == QDialog.Accepted:
            doc = self.current_document()
            if doc.isModified():
                if self.ask_save_changes() == QDialog.Rejected:
                    return QDialog.Rejected

            index = dialog.currentIndex()
            selected = model.item(index)

            self.load_scheme(unicode(selected.path()))
        return status

    def welcome_dialog(self):
        """Show a modal welcome dialog for Orange Canvas.
        """

        dialog = welcomedialog.WelcomeDialog(self)
        dialog.setWindowTitle(self.tr("Welcome to Orange Data Mining"))
        top_row = [self.get_started_action, self.tutorials_action,
                   self.documentation_action]

        def new_scheme():
            if self.new_scheme() == QDialog.Accepted:
                dialog.accept()

        def open_scheme():
            if self.open_scheme() == QDialog.Accepted:
                dialog.accept()

        def open_recent():
            if self.recent_scheme() == QDialog.Accepted:
                dialog.accept()

        new_action = \
            QAction(self.tr("New"), dialog,
                    toolTip=self.tr("Open a new scheme."),
                    triggered=new_scheme,
                    shortcut=QKeySequence.New,
                    icon=canvas_icons("New.svg")
                    )

        open_action = \
            QAction(self.tr("Open"), dialog,
                    objectName="welcome-action-open",
                    toolTip=self.tr("Open a scheme."),
                    triggered=open_scheme,
                    shortcut=QKeySequence.Open,
                    icon=canvas_icons("Open.svg")
                    )

        recent_action = \
            QAction(self.tr("Recent"), dialog,
                    objectName="welcome-recent-action",
                    toolTip=self.tr("Browse and open a recent scheme."),
                    triggered=open_recent,
                    shortcut=QKeySequence(Qt.ControlModifier | \
                                          (Qt.ShiftModifier | Qt.Key_R)),
                    icon=canvas_icons("Recent.svg")
                    )

        self.new_action.triggered.connect(dialog.accept)
        bottom_row = [new_action, open_action, recent_action]

        dialog.addRow(top_row, background="light-grass")
        dialog.addRow(bottom_row, background="light-orange")

        settings = QSettings()

        dialog.setShowAtStartup(
            settings.value("welcomedialog/show-at-startup", True).toBool()
        )

        status = dialog.exec_()

        settings.setValue("welcomedialog/show-at-startup",
                          dialog.showAtStartup())
        return status

    def show_scheme_properties(self):
        """Show current scheme properties.
        """
        dialog = SchemeInfoDialog(self)
        dialog.setWindowTitle(self.tr("Scheme Info"))
        dialog.setFixedSize(725, 450)

        current_doc = self.current_document()
        scheme = current_doc.scheme()
        dialog.setScheme(scheme)
        dialog.exec_()

    def set_canvas_view_zoom(self, zoom):
        doc = self.current_document()
        if zoom:
            doc.view().scale(1.5, 1.5)
        else:
            doc.view().resetTransform()

    def align_to_grid(self):
        "Align widgets on the canvas to an grid."
        self.current_document().alignToGrid()

    def new_arrow_annotation(self):
        """Create and add a new arrow annotation to the current scheme.
        """
        self.current_document().newArrowAnnotation()

    def new_text_annotation(self):
        """Create a new text annotation in the scheme.
        """
        self.current_document().newTextAnnotation()

    def set_signal_freeze(self, freeze):
        scheme = self.current_document().scheme()
        if freeze:
            scheme.signal_manager.freeze().push()
        else:
            scheme.signal_manager.freeze().pop()

    def remove_selected(self):
        """Remove current scheme selection.
        """
        self.current_document().removeSelected()

    def quit(self):
        """Quit the application.
        """
        self.close()

    def undo(self):
        """Undo last action.
        """
        pass

    def redo(self):
        """Redo last action.
        """
        pass

    def select_all(self):
        self.current_document().selectAll()

    def open_widget(self):
        """Open/raise selected widget's GUI.
        """
        self.current_document().openSelected()

    def rename_widget(self):
        """Rename the current focused widget.
        """
        doc = self.current_document()
        nodes = doc.selectedNodes()
        if len(nodes) == 1:
            doc.editNodeTitle(nodes[0])

    def widget_help(self):
        """Open widget help page.
        """
        doc = self.current_document()
        nodes = doc.selectedNodes()
        help_url = None
        if len(nodes) == 1:
            node = nodes[0]
            desc = node.description
            if desc.help:
                help_url = desc.help

        if help_url is not None:
            QDesktopServices.openUrl(QUrl(help_url))
        else:
            message_information(
                self.tr("Sorry there is documentation available for "
                        "this widget."),
                parent=self)

    def open_canvas_settings(self):
        """Open canvas settings/preferences dialog
        """
        pass

    def open_about(self):
        """Open the about dialog.
        """
        dlg = AboutDialog(self)
        dlg.exec_()

    def add_recent_scheme(self, scheme):
        """Add `scheme` to the list of recent schemes.
        """
        if not scheme.path:
            return

        title = scheme.title
        path = scheme.path

        if title is None:
            title = os.path.basename(path)
            title, _ = os.path.splitext(title)

        filename = os.path.abspath(os.path.realpath(path))
        filename = os.path.normpath(filename)

        actions_by_filename = {}
        for action in self.recent_scheme_action_group.actions():
            path = unicode(action.data().toString())
            actions_by_filename[path] = action

        if (title, filename) in self.recent_schemes:
            # Remove the title/filename (so it can be reinserted)
            recent_index = self.recent_schemes.index((title, filename))
            self.recent_schemes.pop(recent_index)

        if filename in actions_by_filename:
            action = actions_by_filename[filename]
            self.recent_menu.removeAction(action)
        else:
            action = QAction(title, self, toolTip=filename)
            action.setData(filename)

        self.recent_schemes.insert(0, (title, filename))

        recent_actions = self.recent_menu.actions()
        begin_index = index(recent_actions, self.recent_menu_begin)
        action_before = recent_actions[begin_index + 1]

        self.recent_menu.insertAction(action_before, action)
        self.recent_scheme_action_group.addAction(action)

        config.save_recent_scheme_list(self.recent_schemes)

    def clear_recent_schemes(self):
        """Clear list of recent schemes
        """
        actions = list(self.recent_menu.actions())

        # Exclude permanent actions (Browse Recent, separators, Clear List)
        actions_to_remove = [action for action in actions \
                             if unicode(action.data().toString())]

        for action in actions_to_remove:
            self.recent_menu.removeAction(action)

        self.recent_schemes = []
        config.save_recent_scheme_list([])

    def _on_recent_scheme_action(self, action):
        """A recent scheme action was triggered by the user
        """
        document = self.current_document()
        if document.isModified():
            if self.ask_save_changes() == QDialog.Rejected:
                return

        filename = unicode(action.data().toString())
        self.load_scheme(filename)

    def _on_dock_location_changed(self, location):
        """Location of the dock_widget has changed, fix the margins
        if necessary.

        """
        self.__update_scheme_margins()

    def closeEvent(self, event):
        """Close the main window.
        """
        document = self.current_document()
        if document.isModified():
            if self.ask_save_changes() == QDialog.Rejected:
                # Reject the event
                event.ignore()
                return

        # Set an empty scheme to clear the document
        document.setScheme(widgetsscheme.WidgetsScheme())
        document.deleteLater()

        config.save_config()

        geometry = self.saveGeometry()
        state = self.saveState(version=self.SETTINGS_VERSION)
        settings = QSettings()
        settings.beginGroup("canvasmainwindow")
        settings.setValue("geometry", geometry)
        settings.setValue("state", state)
        settings.setValue("canvasdock/expanded",
                          self.dock_widget.expanded())
        settings.setValue("scheme_margins_enabled",
                          self.scheme_margins_enabled)

        settings.setValue("last_scheme_dir", self.last_scheme_dir)
        settings.endGroup()

        event.accept()

    def showEvent(self, event):
        settings = QSettings()
        geom_data = settings.value("canvasmainwindow/geometry")
        if geom_data.isValid():
            self.restoreGeometry(geom_data.toByteArray())

        return QMainWindow.showEvent(self, event)

    # Mac OS X
    if sys.platform == "darwin":
        def toggleMaximized(self):
            """Toggle normal/maximized window state.
            """
            if self.isMinimized():
                # Do nothing if window is minimized
                return

            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()

        def changeEvent(self, event):
            if event.type() == QEvent.WindowStateChange:
                # Enable/disable window menu based on minimized state
                self.window_menu.setEnabled(not self.isMinimized())
            QMainWindow.changeEvent(self, event)

    def tr(self, sourceText, disambiguation=None, n=-1):
        """Translate the string.
        """
        return unicode(QMainWindow.tr(self, sourceText, disambiguation, n))


def identity(item):
    return item


def index(sequence, *what, **kwargs):
    """index(sequence, what, [key=None, [predicate=None]])
    Return index of `what` in `sequence`.
    """
    what = what[0]
    key = kwargs.get("key", identity)
    predicate = kwargs.get("predicate", operator.eq)
    for i, item in enumerate(sequence):
        item_key = key(item)
        if predicate(what, item_key):
            return i
    raise ValueError("%r not in sequence" % what)
