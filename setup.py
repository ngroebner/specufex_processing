import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="speufex_preprocessing",
    version="0.1.0",
    author="Specufex team",
    author_email="groe0029@umn.edu",
    description="Preand post processing tools for specufex ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/benholtzman/specufex_preprocessing/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts = [
        "specufex_preprocessing/1_makeWaveformsDataset.py",
        "specufex_preprocessing/2_convertToSpectrograms.py",
        "specufex_preprocessing/2a_plotSpectrum.py",
        "specufex_preprocessing/3_runSpecUFEx.py"
    ],
)