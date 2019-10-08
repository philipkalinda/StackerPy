from distutils.core import setup

# with open('README.md') as f:
#   long_desc = f.read()
    
setup(
  name = 'stackerpy',
  packages = ['stackerpy'],
  version = '0.09',
  license='MIT',
  description = 'Model Stacking for scikit-learn models for Machine Learning (including blending)',
#   long_description=long_desc,
#   long_description_content_type='text/markdown',
  author = 'Philip Kalinda',
  author_email = 'philipkalinda@gmail.com',
  url = 'https://philipkalinda.com/ds10',
  download_url = 'https://github.com/philipkalinda/StackerPy/archive/v_009.tar.gz',
  keywords = ['Model Stacking', 'Blending', 'Stacking', 'Machine Learning', 'Algorithm', 'Scikit-Learn', 'Modelling', 'AI', 'Blending'],
  install_requires=[
          'numpy',
          'pandas',
          'sklearn',
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
