"""

"""
import pkg_resources


# Entry point for main Orange categories/widgets discovery
def widget_discovery(discovery):
    from . import (
        Associate, Classify, Data, Evaluate, Regression,
        Unsupervised, Visualize, Prototypes, VisualizeQt
    )
    dist = pkg_resources.get_distribution("Orange")
    for pkg in [Data, Visualize, Classify, Regression, Evaluate, Unsupervised,
                Associate, Prototypes, VisualizeQt ]:
        discovery.process_category_package(pkg, distribution=dist)


# Intersphinx documentation root's (registered in setup.py)
intersphinx = (
     # root in development mode
     ("{DEVELOP_ROOT}/docs/build/html/", None),
     # URL is taken from PKG-INFO (Home-page)
     ("{URL}/docs/latest/",
      "{URL}/docs/latest/_objects/")
)
