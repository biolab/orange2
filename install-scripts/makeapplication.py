### makeapplication.py                                                          
import bundlebuilder
##bundlebuilder.USE_ZIPIMPORT=None

bundlebuilder.buildapp(
    name='canvas.app',     # what to build                                 
    mainprogram='orange/OrangeCanvas/orngCanvas.py',      # your app's main()                             
    argv_emulation=0,           # drag&dropped filenames show up in sys.argv    
    iconfile='orange.icns',      # file containing your app's icons              
    standalone=1,               # make this app self contained.                 
    includeModules=['sip', 'ConfigParser', 'cPickle', 'cStringIO', 'LinearAlgebra', 'operator', 'math', 'whrandom', 'qtcanvas', 'qttable', 'Numeric', 'qwt', 'xml', 'xml.dom', 'xml.dom.minidom'],          # list of additional Modules to force in        
    includePackages=[],         # list of additional Packages to force in
    additionalPaths=['orange', 'orange/OrangeCanvas', 'orange/OrangeWidgets'],
    libs=['/Developer/qt.old/lib/libqt.3.dylib'],                             # list of shared libs or Frameworks to include       
)

### end of makeapplication.py
