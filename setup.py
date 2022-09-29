
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tnpul',
    version='0.0.1',
    description='Positive unlabeled learning with tensor networks',
    long_description=readme,
    author='Bojan Žunkovič',
    author_email='bojan.zunkovic@fri.uni-lj.si',
    url='https://github.com/qml-tn/pul.git',
    license=license,
    packages=["tnpul"],
)
