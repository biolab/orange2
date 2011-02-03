#/bin/env python
import sys, os

from lib2to3 import refactor, main
from lib2to3.main import diff_texts, StdoutRefactoringTool, warn

#fixes = refactor.get_fixers_from_package("lib2to3.fixes")    

sys.exit(main.main("fixes", sys.argv))




