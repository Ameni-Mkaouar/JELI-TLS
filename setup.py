from setuptools import setup, find_packages
setup(
    name="padlad-tls",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["numpy","scipy","matplotlib","pandas","laspy[lazrs]"],
    entry_points={"console_scripts": ["padlad=cli:main"]},
    author="Ameni Mkaouar, Abdelaziz Kallel",
    description="Joint PAD/LAD estimator from TLS point clouds",
    license="MIT",
)
