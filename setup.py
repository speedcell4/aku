from setuptools import setup, find_packages

name = 'aku'

setup(
    name=name,
    description='An interactive annotation-driven ArgumentParser generator',
    version='0.2.5',
    packages=[package for package in find_packages() if package.startswith(name)],
    url='https://github.com/speedcell4/aku',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    python_requires='>=3.7',
    install_requires=[
    ],
    extras_require={
        'dev': [
            'pytest',
            'hypothesis',
        ],
    }
)
