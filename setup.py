from setuptools import setup, find_packages

setup(
    name="lab2-ml",
    version="0.1.0",
    description="Unsupervised learning & text processing (PCA/t-SNE, KMeans, TF-IDF)",
    author="Victor Ivanov (FF-41mn)",
    packages=find_packages(exclude=("tests",)),
    # встановимо також модуль верхнього рівня main.py (щоб console_script працював)
    py_modules=["main"],
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "pillow",
        "wordcloud",
    ],
    entry_points={
        "console_scripts": [
            # тепер можна запускати просто: `methods --part all`
            "methods=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
