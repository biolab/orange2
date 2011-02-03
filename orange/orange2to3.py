#/bin/env python
import sys, os

from lib2to3 import refactor, main
from lib2to3.main import diff_texts, StdoutRefactoringTool, warn

#fixes = refactor.get_fixers_from_package("lib2to3.fixes")    

dir = os.path.dirname(__file__)
if dir not in sys.path:
    sys.path.insert(0, dir)
sys.exit(main.main("fixes", sys.argv))




