from distutils.core import setup
setup(
  name = 'stackerpy',
  packages = ['stackerpy'],
  version = '0.01',
  license='MIT',
  description = 'Model Stacking for scikit-learn models for Machine Learning',
  author = 'Philip Kalinda',
  author_email = 'philipkalinda@gmail.com',
  url = 'https://philipkalinda.com',
  download_url = 'https://github.com/philipkalinda/stackerpy/archive/v_001.tar.gz',
  keywords = ['Model Stacking', 'Stacking', 'Machine Learning', 'Algorithm', 'Optimization'],
  install_requires=[
          'numpy',
          'sklearn',
		  'geneticfs',
          'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
