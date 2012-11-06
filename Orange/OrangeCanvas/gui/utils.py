"""
Helper utilities

"""
import os
import sys

from contextlib import contextmanager

from PyQt4.QtGui import (
    QWidget, QGradient, QLinearGradient, QRadialGradient, QBrush, QPainter,
    QStyleOption, QStyle
)

import sip


@contextmanager
def updates_disabled(widget):
    """Disable QWidget updates (using QWidget.setUpdatesEnabled)
    """
    old_state = widget.updatesEnabled()
    widget.setUpdatesEnabled(False)
    try:
        yield
    finally:
        widget.setUpdatesEnabled(old_state)


@contextmanager
def signals_disabled(qobject):
    """Disables signals on an instance of QObject.
    """
    old_state = qobject.signalsBlocked()
    qobject.blockSignals(True)
    try:
        yield
    finally:
        qobject.blockSignals(old_state)


def StyledWidget_paintEvent(self, event):
    """A default styled QWidget subclass  paintEvent function.
    """
    opt = QStyleOption()
    opt.init(self)
    painter = QPainter(self)
    self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)


class StyledWidget(QWidget):
    """
    """
    paintEvent = StyledWidget_paintEvent


def is_transparency_supported():
    """Is window transparency supported by the current windowing system.

    """
    if sys.platform == "win32":
        return is_dwm_compositing_enabled()
    elif sys.platform == "cygwin":
        return False
    elif sys.platform == "darwin":
        try:
            # Test if Qt was build against X11.
            from PyQt4.QtGui import QX11Info
            return QX11Info.isCompositingManagerRunning()
        except ImportError:
            # Assuming Quartz compositor is running.
            return True
    elif sys.platform.startswith("linux"):
        # TODO: wayland??
        return is_x11_compositing_enabled()
    elif sys.platform.startswith("freebsd"):
        return is_x11_compositing_enabled()
    elif os.name == "":
        # Any other system (Win, OSX) is assumed to support it
        return True


def is_x11_compositing_enabled():
    """Is X11 compositing manager running.
    """
    try:
        from PyQt4.QtGui import QX11Info
    except ImportError:
        return False

    return QX11Info.isCompositingManagerRunning()


def is_dwm_compositing_enabled():
    """Is Desktop Window Manager compositing (Aero) enabled.
    """
    import ctypes

    enabled = ctypes.c_bool()
    try:
        DwmIsCompositionEnabled = ctypes.windll.dwmapi.DwmIsCompositionEnabled
    except AttributeError:
        # dwmapi or DwmIsCompositionEnabled is not present
        return False

    rval = DwmIsCompositionEnabled(ctypes.byref(enabled))

    return rval == 0 and enabled.value


def gradient_darker(grad, factor):
    """Return a copy of the QGradient darkened by factor.

    .. note:: Only QLinearGradeint and QRadialGradient are supported.

    """
    if type(grad) is QGradient:
        if grad.type() == QGradient.LinearGradient:
            grad = sip.cast(grad, QLinearGradient)
        elif grad.type() == QGradient.RadialGradient:
            grad = sip.cast(grad, QRadialGradient)

    if isinstance(grad, QLinearGradient):
        new_grad = QLinearGradient(grad.start(), grad.finalStop())
    elif isinstance(grad, QRadialGradient):
        new_grad = QRadialGradient(grad.center(), grad.radius(),
                                   grad.focalPoint())
    else:
        raise TypeError

    new_grad.setCoordinateMode(grad.coordinateMode())

    for pos, color in grad.stops():
        new_grad.setColorAt(pos, color.darker(factor))

    return new_grad


def brush_darker(brush, factor):
    """Return a copy of the brush darkened by factor.
    """
    grad = brush.gradient()
    if grad:
        return QBrush(gradient_darker(grad, factor))
    else:
        brush = QBrush(brush)
        brush.setColor(brush.color().darker(factor))
        return brush
