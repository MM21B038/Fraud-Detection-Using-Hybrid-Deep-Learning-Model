
from setuptools import setup, find_packages

setup(
    name='FraudDetectionHybrid',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A hybrid deep learning model for fraud detection.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/YourUsername/FraudDetectionHybrid',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'keras',
    ],
    extras_require={
        'dev': [
            'jupyter',
            'ipython',
            'pytest',
            'flake8',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
