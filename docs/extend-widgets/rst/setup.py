from setuptools import setup

setup(name="Demo",
      packages=["orangedemo"],
      # Declare orangedemo package to contain widgets for the "Demo" category
      entry_points={"orange.widgets": ("Demo = orangedemo")},
      )