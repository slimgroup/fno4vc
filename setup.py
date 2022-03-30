import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

reqs = []
setuptools.setup(
    name="fno4vc",
    version="0.1",
    author="Ali Siahkoohi",
    author_email="alisk@gatech.edu",
    description="Velocity continuation with Fourier neural operators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slimgroup/fno4vc",
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages()
)
