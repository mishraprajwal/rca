from setuptools import setup, find_packages

setup(
    name="rca-system",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "transformers",
        "torch",
        "matplotlib",
        "seaborn",
    ],
    extras_require={
        "dev": ["pytest", "jupyter"],
    },
)