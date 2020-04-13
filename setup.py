from setuptools import setup

setup(
    name='aku',
    description='An Annotation-driven ArgumentParser Generator',
    version='0.2.0',
    packages=['aku'],
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        'Topic :: Utilities',
    ],
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
