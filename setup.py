from setuptools import setup, find_packages

setup(
    name='moccasin',
    packages=['moccasin'],
    package_dir={'moccasin':'src/moccasin'},
    version="0.0.1",
    description="Moccasin, a constraint programming (CP) method for rematerialization",
    # packages=find_packages(exclude=("data","*.egg-info","build")),
    python_requires=">=3.9",
    install_requires=[
        "networkx",
        "matplotlib",
        "pandas",
        "ortools",
        "Jinja2"
    ]
)
