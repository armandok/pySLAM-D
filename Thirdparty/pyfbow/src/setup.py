from setuptools import setup, Extension
from pyDBoW3 import __version__, __short_description__, __author__, __author_email__

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None


def readme():
    with open('../../README.rst') as f:
        return f.read()


setup(name='pyDBoW3',
      version=__version__,
      description=__short_description__,
      long_description=readme(),
      url='http://github.com/foxis/pyDBoW3',
      author=__author__,
      author_email=__author_email__,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Text Processing :: Linguistic',
      ],
      keywords='bag of words bow dbow3 dbow slam orb odometry visual',
      license='MIT',
      packages=['pyDBoW3'],
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'numpy'
      ],
      cmdclass={'bdist_wheel': bdist_wheel},
)