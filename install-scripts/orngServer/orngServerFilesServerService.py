import os
os.chdir(r"C:\Documents and Settings\orngServerFiles\orngServerFiles")

import orngServerFilesServer

import cherrypy

import win32serviceutil
import win32service
import win32event


class MyService(win32serviceutil.ServiceFramework):

    _svc_name_ = "orngServerFilesServerService2"
    _svc_display_name_ = "orngServerFilesServerService2"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        # create an event that SvcDoRun can wait on and SvcStop
        # can set.
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)

    def SvcDoRun(self):
        root, conf = orngServerFilesServer.buildServer()

        #conf["global"]["engine.SIGHUP"] = None
        #conf["global"]["engine.SIGTERM"] = None
        print conf

        cherrypy.config.update(conf)
        cherrypy.tree.mount(root, '/', conf)

        cherrypy.engine.start()
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)
    
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        cherrypy.engine.stop()
        win32event.SetEvent(self.stop_event)

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(MyService)
       

