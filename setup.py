from setuptools import setup, find_packages

setup(
    name="mocassin",
    version="0.0.1",
    description="Checkmate prevents you from OOMing when training big deep neural nets",
    packages=find_packages(exclude=("data","*.egg-info","build")),
    python_requires=">=3.9",
    install_requires=[
        "networkx",
        "matplotlib",
        "pandas",
        "ortools",
        "Jinja2"
    ]
)
