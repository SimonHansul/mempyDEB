from setuptools import setup, find_packages

setup(
    name='mempyDEB',
    version='0.1.0',
    description='DEB-TKTD modelling in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Simon Hansul',
    author_email='simonhansul@gmail.com',
    url='https://github.com/simonhansul/mempyDEB',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'mesa',
        'tqdm'
        ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
