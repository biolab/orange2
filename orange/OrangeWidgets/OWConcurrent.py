"""\
OWConcurent
===========

General helper functions and classes for Orange Canvas
concurent programming
"""
from __future__ import with_statement
from functools import partial

    
from OWWidget import *

class AsyncCall(QObject):
    """ A wrapper class for async function calls using
    Qt's signals for communication with GUI thread

    Arguments:
        - `func`: a function to execute
        - `thread`: a QThread instance to execute under (default `None`,
           threadPool is used instead)
        - `threadPool`: a QThreadPool instance to handle thread allocation
           (for this to work `thread` argument must be `None`)
        - `parent`: parent object (should be None for most cases)
        
    Signals:
        - `starting()`
        - `finished(QString)` - emited when finished with an argument 'OK' on success or repr(ex) on error
        - `finished(PyQt_PyObject, QString)` - same as above but also pass self as an argument 
        - `unhandledException(PyQt_PyObject)` - emited on error with `sys.exc_info()` argument
        - `resultReady(PyQt_PyObject)` - emited on success with function call result as argument
    """
    def __init__(self, func=None, args=(), kwargs={}, thread=None, threadPool=None, parent=None):
        QObject.__init__(self, parent)
        self.func = func
        self._args = args
        self._kwargs = kwargs
        self.threadPool = None
        if thread is not None:
            self.moveToThread(thread)
        else:
            if threadPool is None:
                threadPool = QThreadPool.globalInstance()
            self.threadPool = threadPool
            self._runnable = RunnableAsyncCall(self)
            self.threadPool.start(self._runnable)
            self._connected = False
            return
            
        self.connect(self, SIGNAL("_async_start()"), self.execute, Qt.QueuedConnection)
        self._connected = True


    @pyqtSignature("execute()")
    def execute(self):
        """ Never call directly, use `__call__` or `apply_async` instead
        """
        assert(self.thread() is QThread.currentThread())
        self.emit(SIGNAL("starting()"))
        try:
            self.result  = self.func(*self._args, **self._kwargs)
        except Exception, ex:
            print >> sys.stderr, "Exception in thread ", QThread.currentThread(), " while calling ", self.func
            self.emit(SIGNAL("finished(QString)"), QString(repr(ex)))
            self.emit(SIGNAL("finished(PyQt_PyObject, QString)"), self, QString(repr(ex)))
            self.emit(SIGNAL("unhandledException(PyQt_PyObject)"), sys.exc_info())

            self._exc_info = sys.exc_info()
            self._status = 1
            return

        self.emit(SIGNAL("finished(QString)"), QString("Ok"))
        self.emit(SIGNAL("finished(PyQt_PyObject, QString)"), self, QString("Ok"))
        self.emit(SIGNAL("resultReady(PyQt_PyObject)"), self.result)
        self._status = 0


    def __call__(self, *args, **kwargs):
        """ Apply the call with args and kwargs additional arguments
        """
        if args or kwargs:
            self.func = partial(self.func, *self._args, **self._kwargs)
            self._args, self._kwargs = args, kwargs
            
        if not self._connected:
            QTimer.singleShot(50, self.__call__) # delay until event loop initialized by RunnableAsyncCall
            return
        else:
            self.emit(SIGNAL("_async_start()"))


    def apply_async(self, func, args, kwargs):
        """ call function with `args` as positional and `kwargs` as keyword
        arguments (Overrides __init__ arguments).
        """
        self.func, self._args, self._kwargs = func, args, kwargs
        self.__call__()


    def poll(self):
        """ Return the state of execution.
        """
        return getattr(self, "_status", None)
    
    
    def join(self, processEvents=True):
        """ Wait until the execution finishes.
        """
        while self.poll() is None:
            QThread.currentThread().msleep(50)
            if processEvents and QThread.currentThread() is qApp.thread():
                qApp.processEvents()
                
    def get_result(self, processEvents=True):
        """ Block until the computation completes and return the call result.
        If the execution resulted in an exception, this method will re-raise
        it. 
        """
        self.join(processEvents=processEvents)
        if self.poll() != 0:
            # re-raise the error
            raise self._exc_info[0], self._exc_info[1]
        else:
            return self.result
    
    def emitAdvance(self, count=1):
        self.emit(SIGNAL("advance()"))
        self.emit(SIGNAL("advance(int)"), count)
        
        
    def emitProgressChanged(self, value):
        self.emit(SIGNAL("progressChanged(float)"), value)
        
    
    @pyqtSignature("moveToAndExecute(PyQt_PyObject)")
    def moveToAndExecute(self, thread):
        self.moveToThread(thread)
        
        self.connect(self, SIGNAL("_async_start()"), self.execute, Qt.QueuedConnection)
        
        self.emit(SIGNAL("_async_start()"))
        
        
    @pyqtSignature("moveToAndInit(PyQt_PyObject)")
    def moveToAndInit(self, thread):
        self.moveToThread(thread)
        
        self.connect(self, SIGNAL("_async_start()"), self.execute, Qt.QueuedConnection)
        self._connected = True
        

class WorkerThread(QThread):
    """ A worker thread
    """
    def run(self):
        self.exec_()
        
        
class RunnableTask(QRunnable):
    """ Wrapper for an AsyncCall
    """
    def __init__(self, call):
        QRunnable.__init__(self)
        self.setAutoDelete(False)
        self._call = call
        
    def run(self):
        if isinstance(self._call, AsyncCall):
            self.eventLoop = QEventLoop()
            self.eventLoop.processEvents()
            QObject.connect(self._call, SIGNAL("finished(QString)"), lambda str: self.eventLoop.quit())
            QMetaObject.invokeMethod(self._call, "moveToAndInit", Qt.QueuedConnection, Q_ARG("PyQt_PyObject", QThread.currentThread()))
            self.eventLoop.processEvents()
            self.eventLoop.exec_()
        else:
            self._return = self._call()
            
            
class RunnableAsyncCall(RunnableTask):
    def run(self):
        self.eventLoop = QEventLoop()
        self.eventLoop.processEvents()
        QObject.connect(self._call, SIGNAL("finished(QString)"), lambda str: self.eventLoop.quit())
        QMetaObject.invokeMethod(self._call, "moveToAndInit", Qt.QueuedConnection, Q_ARG("PyQt_PyObject", QThread.currentThread()))
        self.eventLoop.processEvents()
        self.eventLoop.exec_()

def createTask(call, args=(), kwargs={}, onResult=None, onStarted=None, onFinished=None, onError=None, thread=None, threadPool=None):
    async = AsyncCall(thread=thread, threadPool=threadPool)
    if onResult is not None:
        async.connect(async, SIGNAL("resultReady(PyQt_PyObject)"), onResult, Qt.QueuedConnection)
    if onStarted is not None:
        async.connect(async, SIGNAL("starting()"), onStarted, Qt.QueuedConnection)
    if onFinished is not None:
        async.connect(async, SIGNAL("finished(QString)"), onFinished, Qt.QueuedConnection)
    if onError is not None:
        async.connect(async, SIGNAL("unhandledException(PyQt_PyObject)"), onError, Qt.QueuedConnection)
    async.apply_async(call, args, kwargs)
    return async
        
from functools import partial
        
class ProgressBar(QObject):
    """ A thread safe progress callback using Qt's signal mechanism
    to deliver progress updates to the GUI thread. Make sure this object instance
    is created in the GUI thread or is a child of an object from the GUI thread
    """
    
    def __init__(self, widget, iterations, parent=None):
        QObject.__init__(self, parent)
        assert (qApp.thread() is self.thread())
        self.iter = iterations
        self.widget = widget
        self.count = 0
        self.widget.progressBarInit()
        
    def advance(self, count=1):
        self.count += count
        value = int(self.count*100/self.iter)
        QMetaObject.invokeMethod(self, "queuedInvoke", Qt.QueuedConnection, Q_ARG("PyQt_PyObject", partial(self.widget.progressBarSet, value)))

    def finish(self):
        QMetaObject.invokeMethod(self, "queuedInvoke", Qt.QueuedConnection, Q_ARG("PyQt_PyObject", self.widget.progressBarFinished))
        
    def progressBarSet(self, value):
        QMetaObject.invokeMethod(self, "queuedInvoke", Qt.QueuedConnection, Q_ARG("PyQt_PyObject", partial(self.widget.progressBarSet, value)))
    
    @pyqtSignature("queuedInvoke(PyQt_PyObject)")
    def queuedInvoke(self, func):
        func()
        
        
class synchronized(object):
    def __init__(self, object, mode=QMutex.Recursive):
        if not hasattr(object, "_mutex"):
            object._mutex = QMutex(mode)
        self.mutex = object._mutex
        
    def __enter__(self):
        self.mutex.lock()
        return self
    
    def __exit__(self, exc_type=None, exc_value=None, tb=None):
        self.mutex.unlock()

_global_thread_pools = {}
        
        
def threadPool(self, class_="global", private=False, maxCount=None):
    with synchronized(threadPool):
        if private:
            pools = self._private_thread_pools
        else:
            pools = _global_thread_pools
            
        if class_ not in pools:
            if class_ == "global":
                instance = QThreadPool.globalInstance()
            else:
                instance = QThreadPool()
                instance.setMaxThreadCount(maxCount)
            pools[class_] = instance
        return pools[class_]
    
OWBaseWidget.threadPool = threadPool
        

"""\
A multiprocessing like API
==========================
"""

class Process(AsyncCall):
    _process_id = 0
    def __init__(group=None, target=None, name=None, args=(), kwargs={}):
        self.worker = WorkerThread()
        AsyncCall.__init__(self, thread=self.worker)
        
        self.conenct(self, SIGANL("finished(QString)"), self.onFinished, Qt.QueuedConnection)
        self.connect(self, SIGNAL("finished(QString)"), lambda:self.worker.quit(), Qt.QueuedConnection)
        self.target = target
        self.args = args
        self.kwargs = kwargs
        if name is None:
            self.name = "Process-%i" % self._process_id
            Process._process_id += 1
        else:
            self.name = name
        self.exitcode = -1
            
    def start(self):
        self.worker.start()
        self.async_call(self.run)

    def run(self):
        self._result = self.target(*self.args, **self.kwargs)
         
    def join(self):
        while self.poll() is None:
            time.sleep(10)

    def is_alive(self):
        return self.poll() is None
    
    def onFinished(self, string):
        self.exitcode = self._status
        
    def terminate(self):
        self.worker.terminate()
    
from Queue import Queue

class Pool(QObject):
    def __init__(self, processes=None):
        if processes is None:
            import multiprocessing
            processes = multiprocessing.cpu_count()
        self.processes = processes
        self.pool = [Process() for i in range(processes)]
        self._i = 0
    def get_process(self):
        self._i = (self._i + 1) % len(self.pool)
        return self.pool[self._i]
     
    def apply_async(func, args, kwargs):
        process = self.get_process()
        process.start()
        
    def start(self, ):
        pass

