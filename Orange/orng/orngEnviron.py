import Orange.misc.environ as environ

directoryNames = dict(
    orangeDir = environ.install_dir,
    orangeDocDir = environ.doc_install_dir,
    orangeVer = environ.version,
    canvasDir = environ.canvas_install_dir,
    widgetDir = environ.widget_install_dir,
    picsDir = environ.icons_install_dir,
    addOnsDirSys = environ.add_ons_dir,
    addOnsDirUser = environ.add_ons_dir_user,
    applicationDir = environ.application_dir,
    outputDir = environ.output_dir,
    defaultReportsDir = environ.default_reports_dir,
    orangeSettingsDir = environ.orange_settings_dir,
    widgetSettingsDir = environ.widget_settings_dir,
    canvasSettingsDir = environ.canvas_settings_dir,
    bufferDir = environ.buffer_dir
    )
globals().update(directoryNames)

samepath = environ.samepath
addOrangeDirectoriesToPath = environ.add_orange_directories_to_path
