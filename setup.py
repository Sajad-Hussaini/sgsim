from setuptools import setup, find_packages

DESCRIPTION = "SGSIM: a site-based stochastic ground motion simulation model."
LONG_DESCRIPTION = """SGSIM: a site-based stochastic ground motion simulation model.
This model can be used to simulate earthquake ground motions for specific earthquake and site characteristics."""

setup(
    name="SGSIM",
    version="1.0.0",
    author="S.M.Sajad Hussaini",
    author_email="hussaini.smsajad@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['numpy==2.1.2',
        'scipy==1.14.1',
        'matplotlib==3.9.2',
        'pandas==2.2.3'],
    keywords=['python', 'SGSIM', 'stochastic', 'ground motion', 'simulation', 'site-based'],
    include_package_data=True,
    package_data={
        'SGSIM.examples.real_records': ['*'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CC BY-NC-SA 4.0",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Engineering/Research",
    ]
    )
