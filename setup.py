from setuptools import setup, find_packages

setup(
    name="geoinverse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    author="Chen Weifeng",
    description="A geophysical inversion package for electromagnetic methods",
    python_requires=">=3.6"
)