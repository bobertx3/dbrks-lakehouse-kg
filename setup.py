from setuptools import setup, find_packages

setup(
    name="lakehouse-kg",
    version="0.1.0",
    description="Lakehouse Knowledge Graph Starter Kit - Agentic Triplet Generation on Databricks",
    packages=find_packages(include=["lakehouse_kg", "lakehouse_kg.*"]),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "pyspark>=3.4.0",
        "scikit-learn>=1.2.0",
        "networkx>=3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "llm": [
            "databricks-sdk>=0.20.0",
            "openai>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
)
