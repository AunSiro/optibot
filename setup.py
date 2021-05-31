import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optibot", # Replace with your own username
    version="0.0.1",
    author="Siro Moreno",
    author_email="siro.moreno.martin@edu.upc",
    description="Small package containing models and functions for using lagrangian mechanics in order to optimize robot trajectories and control“,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AunSiro/optibot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)