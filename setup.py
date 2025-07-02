from setuptools import setup, find_packages

setup(
    name="menipy",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=["main"],
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=1.5.3",
        "scikit-image>=0.19.3",
        "opencv-python>=4.7.0.72",
        "PySide6>=6.5.1",
    ],
    entry_points={"console_scripts": ["menipy=main:main"]},
)