import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optibot", # Replace with your own username
    version="0.0.1",
    author="Siro Moreno",
    author_email="siro.moreno.martin@upc.edu",
    description="Small package containing models and functions for using collocation methods for trajectory optimization and control",
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
