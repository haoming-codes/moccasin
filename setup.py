from setuptools import setup

setup(
    name="mocassin",
    version="0.0.1",
    description="Checkmate prevents you from OOMing when training big deep neural nets",
    packages=["mocassin"],  # find_packages()
    python_requires=">=3.9",
    install_requires=[
        "networkx",
        "matplotlib",
        "pandas",
        "ortools",
        "Jinja2"
    ]
)
