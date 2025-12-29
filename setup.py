"""
Setup script for JOOLA Customer Analytics package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="joola-customer-analytics",
    version="1.0.0",
    author="Next Gen Consulting",
    author_email="samantha.yung@columbia.edu",
    description="Customer analytics and churn prediction for JOOLA pickleball",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/joola-customer-analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.28.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
