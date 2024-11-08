from setuptools import setup, find_packages

setup(
    name='guerilla',
    version='1.0.5',
    author='Xiaohui Peng',
    author_email='xiaohuipeng.cn@gmail.com',
    description='A package for quantitative trading',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package requires
        'akshare>=1.14.27',
        'alphalens>=0.4.0',
        'jsonpath>=0.82.2',
        'statsmodels>=0.14.0',
        'ta>=0.10.2',
        'yfinance>=0.2.31',
        'pandas-datareader>=0.10.0'
    ],
)