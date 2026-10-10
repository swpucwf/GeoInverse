from setuptools import setup, find_packages

setup(
    name="geoinverse",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.12",
        "matplotlib>=3.5.0",
        "discretize>=0.12",
        "pymatsolver>=0.3",
        "geoana>=0.8.1",
        "libdlf",
    ],
    author="Chen Weifeng",
    description="A geophysical inversion package for electromagnetic methods",
    python_requires=">=3.10"
)