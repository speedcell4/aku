from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fp:
    long_description = fp.read()

setup(
    name='aku',
    description='Annotation-driven ArgumentParser Generator',
    long_description=long_description,
    version=open('VERSION', mode='r').read(),
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.7',
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
    python_requires='>=3.7',
)
