#!usr/bin/env python
"""
A setup script to use setuptools. Can build python eggs and .mpkg using
bdist_mpkg on Mac OSX.

"""
import sys
from setuptools import setup
import distutils.core
distutils.core.have_setuptools = True

execfile("setup.py")