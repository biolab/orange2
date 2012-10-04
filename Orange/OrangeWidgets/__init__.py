"""

"""
import pkg_resources


# Entry point for main Orange categories/widgets discovery
def widget_discovery(discovery):
    from . import (
        Associate, Classify, Data, Evaluate, Regression,
        Unsupervised, Visualize, Prototypes
    )
    dist = pkg_resources.get_distribution("Orange")
    for pkg in [Data, Visualize, Classify, Regression, Evaluate, Unsupervised,
                Associate, Prototypes]:
        discovery.process_category_package(pkg, distribution=dist)
