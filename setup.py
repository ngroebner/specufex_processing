import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="specufex-processing",
    version="0.2.0",
    author="Specufex team",
    author_email="groe0029@umn.edu",
    description="Pre and post processing tools for specufex ",
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
        "scripts/1_makeWaveformsDataset.py",
        "scripts/2_convertToSpectrograms.py",
        "scripts/2a_plotSpectrum.py",
        "scripts/3_runSpecUFEx.py"
    ],
)
