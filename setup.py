import setuptools

long_description='experimental version'

setuptools.setup(
    name="chainforge",
    version="0.0.01",
    license="MIT",
    author="Ravil Dorozhinskii",
    author_email="ravil.aviva.com@gmail.com",
    description="GPU-GEMM generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/ravil-mobile/chainforge/wiki",
    python_requires='>=3.5',
    #install_requires=install_requires,
    #include_package_data=True,
)
