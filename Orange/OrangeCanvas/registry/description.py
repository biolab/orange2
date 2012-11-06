"""
Widget meta description classes.

"""

import os
import sys


# Exceptions

class DescriptionError(Exception):
    pass


class WidgetSpecificationError(DescriptionError):
    pass


class SignalSpecificationError(DescriptionError):
    pass


class CategorySpecificationError(DescriptionError):
    pass


##############
# Signal flags
##############

# A single signal
Single = 2

# Multiple signal (more then one input on the channel)
Multiple = 4

# Default signal (default or primary input/output)
Default = 8
NonDefault = 16

# Explicit - only connected if specifically requested or the only possibility
Explicit = 32

# Dynamic type output signal
Dynamic = 64


# Input/output signal (channel) description


class InputSignal(object):
    """Description of an input channel.

    Parameters
    ----------
    name : str
        Name of the channel.
    type : str or `type`
        Type of the accepted signals.
    handler : str
        Name of the handler method for the signal.
    parameters : int
        Parameter flags.

    """
    def __init__(self, name, type, handler, parameters=Single + NonDefault):
        self.name = name
        self.type = type
        self.handler = handler

        if isinstance(parameters, basestring):
            # parameters are stored as strings
            parameters = eval(parameters)

        if not (parameters & Single or parameters & Multiple):
            parameters += Single

        if not (parameters & Default or parameters & NonDefault):
            parameters += NonDefault

        self.single = parameters & Single
        self.default = parameters & Default
        self.explicit = parameters & Explicit


class OutputSignal(object):
    """Description of an output channel.

    Parameters
    ----------
    name : str
        Name of the channel.
    type : str or `type`
        Type of the output signals.
    parameters : int
        Parameter flags.

    """
    def __init__(self, name, type, parameters=Single + NonDefault):
        self.name = name
        self.type = type

        if isinstance(parameters, basestring):
            # parameters are stored as strings
            parameters = eval(parameters)

        if not (parameters & Single or parameters & Multiple):
            parameters += Single

        if not (parameters & Default or parameters & NonDefault):
            parameters += NonDefault

        self.single = parameters & Single
        self.default = parameters & Default
        self.explicit = parameters & Explicit

        self.dynamic = parameters & Dynamic
        if self.dynamic and not self.single:
            raise SignalSpecificationError(
                "Output signal can not be 'Multiple' and 'Dynamic'."
                )


class WidgetDescription(object):
    """Description of a widget.

    Parameters
    ==========
    name : str
        A human readable name of the widget (required).
    category : str
        A name of the category in which this widget belongs (optional).
    version : str
        Version of the widget (optional).
    description : str
        A short description of the widget, suitable for a tool tip (optional).
    long_description : str
        A longer description of the widget (optional).
    quialified_name : str
        A qualified name (import name) of the class implementation (required).
    package : str
        A package name where the widget is implemented (optional).
    project_name : str
        The distribution name that provides the widget (optional)
    inputs : list of `InputSignal`
        A list of input channels provided by the widget.
    outputs : list of `OutputSignal`
        A list of output channels provided the widget.
    help : str
        URL or an URL scheme of a detailed widget help page.
    author : str
        Author name.
    author_email : str
        Author email address.
    maintainer : str
        Maintainer name
    maintainer_email : str
        Maintainer email address.
    keywords : str
        A comma separated list of keyword phrases.
    priority : int
        Widget priority (the order of the widgets in a GUI presentation).
    icon : str
        A filename of the widget icon (in relation to the package).
    background : str
        Widget's background color (in the canvas GUI).

    """
    def __init__(self, name=None, category=None, version=None,
                 description=None, long_description=None,
                 qualified_name=None, package=None, project_name=None,
                 inputs=[], outputs=[], help=None,
                 author=None, author_email=None,
                 maintainer=None, maintainer_email=None,
                 url=None, keywords=None, priority=sys.maxint,
                 icon=None, background=None,
                 ):

        if not qualified_name:
            # TODO: Should also check that the name is real.
            raise ValueError("'qualified_name' must be supplied.")

        self.name = name
        self.category = category
        self.version = version
        self.description = description
        self.long_description = long_description
        self.qualified_name = qualified_name
        self.package = package
        self.project_name = project_name
        self.inputs = inputs
        self.outputs = outputs
        self.help = help
        self.author = author
        self.author_email = author_email
        self.maintainer = maintainer
        self.maintainer_email = maintainer_email
        self.url = url
        self.keywords = keywords
        self.priority = priority
        self.icon = icon
        self.background = background

    def __str__(self):
        return "WidgetDescription(name=%(name)r, category=%(category)r, ...)" \
                % self.__dict__

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_file(cls, filename, import_name=None):
        """Widget description from old style (2.5 version) widget
        descriptions.

        """
        from Orange.orng.widgetParser import WidgetMetaData
        from ..orngSignalManager import resolveSignal

        rest, ext = os.path.splitext(filename)
        if ext in [".pyc", ".pyo"]:
            filename = filename[:-1]

        contents = open(filename, "rb").read()

        dirname, basename = os.path.split(filename)
        default_cat = os.path.basename(dirname)

        try:
            meta = WidgetMetaData(contents, default_cat)
        except Exception, ex:
            if "Not an Orange widget module." in str(ex):
                raise WidgetSpecificationError
            else:
                raise

        widget_name, ext = os.path.splitext(basename)
        if import_name is None:
            import_name = widget_name

        wmod = __import__(import_name, fromlist=[""])

        inputs = eval(meta.inputList)
        outputs = eval(meta.outputList)

        inputs = [InputSignal(*input) for input in inputs]
        outputs = [OutputSignal(*output) for output in outputs]

        # Resolve signal type names into concrete type instances
        inputs = [resolveSignal(input, globals=wmod.__dict__)
                  for input in inputs]
        outputs = [resolveSignal(output, globals=wmod.__dict__)
                  for output in outputs]

        # Convert all signal types back into qualified names.
        # This is to prevent any possible import problems when cached
        # descriptions are unpickled (the relevant code using this lists
        # should be able to handle missing types better).
        for s in inputs + outputs:
            s.type = "%s.%s" % (s.type.__module__, s.type.__name__)

        desc = WidgetDescription(
             name=meta.name,
             category=meta.category,
             description=meta.description,
             qualified_name="%s.%s" % (import_name, widget_name),
             package=wmod.__package__,
             keywords=meta.tags,
             inputs=inputs,
             outputs=outputs,
             icon=meta.icon,
             priority=int(meta.priority)
            )
        return desc

    @classmethod
    def from_module(cls, module):
        """Get the widget description from a module.

        The module is inspected for global variables (upper case versions of
        `WidgetDescription.__init__` parameters).

        Parameters
        ----------
        module : `module` or str
            A module to inspect for widget description. Can be passed
            as a string (qualified import name).

        """
        if isinstance(module, basestring):
            module = __import__(module, fromlist=[""])

        module_name = module.__name__.rsplit(".", 1)[-1]
        if module.__package__:
            package_name = module.__package__.rsplit(".", 1)[-1]
        else:
            package_name = None

        # Default widget class name unless otherwise specified is the
        # module name, and default category the package name
        default_cls_name = module_name
        default_cat_name = package_name if package_name else ""

        widget_cls_name = getattr(module, "WIDGET_CLASS", default_cls_name)
        try:
            widget_class = getattr(module, widget_cls_name)
            name = getattr(module, "NAME")
        except AttributeError:
            # The module does not have a widget class implementation or the
            # widget name.
            raise WidgetSpecificationError

        inputs = getattr(module, "INPUTS", [])
        outputs = getattr(module, "OUTPUTS", [])
        category = getattr(module, "CATEGORY", default_cat_name)
        description = getattr(module, "DESCRIPTION", name)

        qualified_name = "%s.%s" % (module.__name__, widget_class.__name__)

        icon = getattr(module, "ICON", None)
        priority = getattr(module, "PRIORITY", sys.maxint)
        keywords = getattr(module, "KEYWORDS", None)

        inputs = [InputSignal(*t) for t in inputs]
        outputs = [OutputSignal(*t) for t in outputs]

        # Convert all signal types into qualified names.
        # This is to prevent any possible import problems when cached
        # descriptions are unpickled (the relevant code using this lists
        # should be able to handle missing types better).
        for s in inputs + outputs:
            s.type = "%s.%s" % (s.type.__module__, s.type.__name__)

        return WidgetDescription(
            name=name,
            category=category,
            description=description,
            qualified_name=qualified_name,
            package=module.__package__,
            inputs=inputs,
            outputs=outputs,
            icon=icon,
            priority=priority,
            keywords=keywords)


class CategoryDescription(object):
    """Description of a widget category.

    Parameters
    ==========

    name : str
        A human readable name.
    version : str
        Version string (optional).
    description : str
        A short description of the category, suitable for a tool
        tip (optional).
    long_description : str
        A longer description.
    qualified_name : str
        Qualified name
    project_name : str
        A project name providing the category.
    priority : int
        Priority (order in the GUI).
    icon : str
        An icon filename
    background : str
        An background color for widgets in this category.

    """
    def __init__(self, name=None, version=None,
                 description=None, long_description=None,
                 qualified_name=None, package=None,
                 project_name=None, help=None,
                 author=None, author_email=None,
                 maintainer=None, maintainer_email=None,
                 url=None, keywords=None,
                 widgets=None, priority=sys.maxint,
                 icon=None, background=None
                 ):

        self.name = name
        self.version = version
        self.description = description
        self.long_description = long_description
        self.qualified_name = qualified_name
        self.package = package
        self.project_name = project_name
        self.help = help
        self.author = author
        self.author_email = author_email
        self.maintainer = maintainer
        self.maintainer_email = maintainer_email
        self.url = url
        self.keywords = keywords
        self.widgets = widgets or []
        self.priority = priority
        self.icon = icon
        self.background = background

    def __str__(self):
        return "CategoryDescription(name=%(name)r, ...)" % self.__dict__

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_package(cls, package):
        """Get the CategoryDescription from a package.

        Parameters
        ----------
        package : `module` or `str`
            A package containing the category.

        """
        if isinstance(package, basestring):
            package = __import__(package, fromlist=[""])
        package_name = package.__name__
        qualified_name = package_name
        default_name = package_name.rsplit(".", 1)[-1]

        name = getattr(package, "NAME", default_name)
        description = getattr(package, "DESCRIPTION", None)
        long_description = getattr(package, "LONG_DESCRIPTION", None)
        help = getattr(package, "HELP", None)
        author = getattr(package, "AUTHOR", None)
        author_email = getattr(package, "AUTHOR_EMAIL", None)
        maintainer = getattr(package, "MAINTAINER", None)
        maintainer_email = getattr(package, "MAINTAINER_MAIL", None)
        url = getattr(package, "URL", None)
        keywords = getattr(package, "KEYWORDS", None)
        widgets = getattr(package, "WIDGETS", None)
        priority = getattr(package, "PRIORITY", sys.maxint - 1)
        icon = getattr(package, "ICON", None)
        background = getattr(package, "BACKGROUND", None)

        if priority == sys.maxint - 1 and name.lower() == "prototypes":
            priority = sys.maxint

        return CategoryDescription(
            name=name,
            qualified_name=qualified_name,
            description=description,
            long_description=long_description,
            help=help,
            author=author,
            author_email=author_email,
            maintainer=maintainer,
            maintainer_email=maintainer_email,
            url=url,
            keywords=keywords,
            widgets=widgets,
            priority=priority,
            icon=icon,
            background=background)
