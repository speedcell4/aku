from setuptools import find_packages, setup

name = 'aku'

setup(
    name=name,
    description='An interactive annotation-driven ArgumentParser generator',
    version='0.2.6',
    packages=[package for package in find_packages() if package.startswith(name)],
    url='https://github.com/speedcell4/aku',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    python_requires='>=3.8',
)
