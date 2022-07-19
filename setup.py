import setuptools

with open('README.md', 'r') as readme:
    description = readme.read()

setuptools.setup(
    name='shorthand',
    version='0.1.1',
    author='Devin Short',
    author_email='short.devin@gmail.com',
    packages=['shorthand'],
    description=(
        'A Python package for parsing text data according to custom '
        'syntax files'
    ),
    long_description=description,
    long_description_content_type="text/markdown",
    url='https://github.com/shortorian/shorthand',
    license='MIT',
    python_requires='>=3',
    install_requires=[]
)