from setuptools import setup, find_packages

setup(
    name="SGSIM",
    version="1.0.0",
    author="S.M.Sajad Hussaini",
    author_email="hussaini.smsajad@gmail.com",
    description="SGSIM: stochastic ground motion simulation model.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sajad-Hussaini/SGSIM",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=['numpy', 'scipy', 'numba', 'pandas', 'matplotlib', 'h5py'],
    keywords=['Python', 'SGSIM', 'simulation model', 'stochastic ground motion'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU AGPL v3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Engineering/Research"]
    )
