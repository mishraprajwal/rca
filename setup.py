from setuptools import setup, find_packages

setup(
    name="rca-system",
    version="0.2.0",
    description="Automated hierarchical root-cause analysis for IT incident tickets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="RCA Contributors",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "nltk>=3.8",
        "joblib>=1.3",
        "PyYAML>=6.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "transformer": [
            "transformers>=4.33",
            "torch>=2.0",
        ],
        "viz": [
            "matplotlib>=3.7",
            "seaborn>=0.12",
        ],
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rca-train=src.__main__:main",
            "rca-predict=predict:main",
        ],
    },
)
