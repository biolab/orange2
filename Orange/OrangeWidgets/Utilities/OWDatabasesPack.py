from __future__ import division

import os
import warnings
import tarfile
import urlparse
import posixpath

from PyQt4.QtGui import QProgressBar

from PyQt4.QtCore import Qt, QUrl, QTimer
from PyQt4.QtNetwork import (
    QNetworkAccessManager, QNetworkRequest, QNetworkReply
)

from Orange.OrangeWidgets import OWWidget, OWGUI
from Orange.utils import serverfiles, environ

NAME = "Databases Package"
DESCRIPTION = "Download a bunch of files in one step."
ICON = "icons/DatabasesPkg.svg"


class OWDatabasesPack(OWWidget.OWWidget):
    settingsList = ["fileslist", "downloadurl", "downloadmessage"]

    def __init__(self, parent=None, signalManager=None,
                 title="Databases Pack"):

        super(OWDatabasesPack, self).__init__(
            parent, signalManager, title, wantMainArea=False)

        self.fileslist = [("Taxonomy", "ncbi_taxonomy.tar.gz")]
        self.downloadurl = "https://dl.dropboxusercontent.com/u/100248799/sf_pack.tar.gz"
        self.downloadmessage = (
            "Downloading a subset of available databases for a smoother " +
            "ride through the workshop"
        )
        self.loadSettings()
        self._tmpfile = None
        self.reply = None

        # Locks held on server files
        self.locks = []
        self.net_manager = QNetworkAccessManager()
        # Lock all files in files list so any other (in process) atempt to
        # download the files waits)
        box = OWGUI.widgetBox(self.controlArea, "Info")
        self.info = OWGUI.widgetLabel(box, self.downloadmessage)
        self.info.setWordWrap(True)

        box = OWGUI.widgetBox(self.controlArea, "Status")
        self.statusinfo = OWGUI.widgetLabel(box, "Please wait")
        self.statusinfo.setWordWrap(True)
        self.progressbar = QProgressBar()
        box.layout().addWidget(self.progressbar)

        self.setMinimumWidth(250)

        already_available = [(domain, filename)
                             for domain in serverfiles.listdomains()
                             for filename in serverfiles.listfiles(domain)]

        if set(self.fileslist) <= set(already_available):
            # All files are already downloaded
            self.statusinfo.setText("All files already available")
        else:
            for domain, filename in self.fileslist + [("_tmp_cache_", "pack")]:
                manager = serverfiles._lock_file(domain, filename)
                try:
                    manager.__enter__()
                except Exception:
                    warnings.warn("Could not acquire lock for {0} {0}"
                                  .format(domain, filename))
                    self.warning(0, "...")
                else:
                    self.locks.append(manager)

            QTimer.singleShot(0, self.show)
            QTimer.singleShot(0, self.run)

    def run(self):
        self.info.setText(self.downloadmessage)

        target_dir = os.path.join(environ.buffer_dir, "serverfiles_pack_cache")

        try:
            os.makedirs(target_dir)
        except OSError:
            pass

        self.progressBarInit()

        req = QNetworkRequest(QUrl(self.downloadurl))
        self.reply = self.net_manager.get(req)
        self.reply.downloadProgress.connect(self._progress)
        self.reply.error.connect(self._error)
        self.reply.finished.connect(self._finished)
        self.reply.readyRead.connect(self._read)
        url = urlparse.urlsplit(self.downloadurl)
        self._tmpfilename = posixpath.basename(url.path)

        self._tmpfile = open(
            os.path.join(target_dir, self._tmpfilename), "wb+"
        )

    def _progress(self, rec, total):
        if rec == 0 and total == 0:
            self.progressbar.setRange(0, 0)
            self.progressBarValue = 1.0
        else:
            self.progressbar.setRange(0, total)
            self.progressbar.setValue(rec)
            self.progressBarValue = 100.0 * rec / (total or 1)

    def _error(self, error):
        self.statusinfo.setText(
            u"Error: {0}".format(error.errorString())
        )
        self.error(0, "Error: {0}".format(error.errorString()))
        self._releaselocks()
        self._removetmp()
        self.progressBarFinshed()

    def _read(self):
        contents = str(self.reply.readAll())
        self._tmpfile.write(contents)

    def _finished(self):
        self.progressbar.reset()
        if self.reply.error() != QNetworkReply.NoError:
            self._releaselocks()
            self._removetmp()
            return

        self.statusinfo.setText("Extracting")
        try:
            self._extract()
        except Exception:
            # Permission errors, ... ??
            pass

        self.statusinfo.setText("Done")
        self.progressbar.reset()
        self.progressBarFinished()

        self._releaselocks()
        self._removetmp()

    def _extract(self):
        self._tmpfile.seek(0, 0)
        archive = tarfile.open(fileobj=self._tmpfile)
        target_dir = serverfiles.localpath()
        archive.extractall(target_dir)

    def onDeleteWidget(self):
        super(OWDatabasesPack, self).onDeleteWidget()
        self._releaselocks()
        if self._tmpfile is not None:
            self._tmpfile.close()
            self._tmpfile = None

    def _releaselocks(self):
        for lock in self.locks:
            lock.__exit__(None, None, None)
        self.locks = []

    def _removetmp(self):
        if self._tmpfile is not None:
            self._tmpfile.close()
            self._tmpfile = None

        tmp_file = os.path.join(
            environ.buffer_dir, "serverfiles_pack_cache", self._tmpfilename
        )

        os.remove(tmp_file)
