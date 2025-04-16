from setuptools import setup, find_packages

setup(
    name="sgsim",
    version="1.0.61",
    author="S.M.Sajad Hussaini",
    author_email="hussaini.smsajad@gmail.com",
    description="SGSIM: Stochastic Ground-motion SIMulation model.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sajad-Hussaini/sgsim",
    license="AGPL-3.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=['numpy', 'scipy', 'numba', 'matplotlib'],
    keywords=['Python', 'SGSIM', 'stochastic ground motion simulation', 'earthquake engineering'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU AGPL v3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Engineering/Research",
        "Topic :: Scientific/Engineering",]
    )
