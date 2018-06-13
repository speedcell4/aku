from setuptools import setup

setup(
    name='aku',
    description='let your ideas take flight, aku made (飽くまで)',
    version=open('VERSION', mode='r').read(),
    packages=['aku'],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        'Topic :: Utilities',
    ],
    url='https://github.com/speedcell4/aku',
    license='MIT',
    author='Izen',
    author_email='speedcell4@gmail.com',
)
