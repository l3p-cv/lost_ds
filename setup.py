import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

# read required packages from requirements.txt
# thelibFolder = os.path.dirname(os.path.realpath(__file__))
# requirementPath = thelibFolder + '/requirements.txt'
requirementPath = 'requirements.txt'
install_requires = [] 
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="lost_ds", # Replace with your own username
    version="donotchange",
    author="L3bm GmbH",
    author_email="info@l3bm.com",
    description="Lost Dataset library",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/l3p-cv/lost_ds",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)