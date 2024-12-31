from setuptools import setup, find_packages

setup(
    name="SGSIM",
    version="1.0.0",
    author="S.M.Sajad Hussaini",
    author_email="hussaini.smsajad@gmail.com",
    description="SGSIM: a site-based stochastic ground motion simulation model.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sajad-Hussaini/SGSIM",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=['numpy', 'scipy', 'numba', 'pandas', 'matplotlib'],
    keywords=['Python', 'SGSIM', 'stochastic', 'ground motion', 'simulation model',
              'site-based', 'stochastic ground motion', 'near-fault', 'multiple strong phase'
              'directivity pulse', 'basin effect'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU AGPL v3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Engineering/Research"]
    )
