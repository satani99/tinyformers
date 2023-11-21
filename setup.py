from setuptools import setup, find_packages 

setup(
    name = 'x-transformers-tinygrad',
    packages = find_packages(exclude=['examples']),
    version = 'v0.2.0-alpha',
    license = 'MIT',
    description = 'X-Transformers - Tinygrad',
    author = 'Nikhil Satani',
    author_email = 'sataninikhil@gmail.com',
    url = 'https://github.com/satani99/x_transformers_tinygrad',
    long_description_content_type = 'text/markdown',
    keyword = [
        'artificial intelligence',
        'attention mechanism',
        'transformers',
        'tinygrad'
    ],
    install_requires=[
        'tinygrad>=0.7.0',
        'einops>=0.7.0'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Laguage :: Python :: 3.10',
    ],
)