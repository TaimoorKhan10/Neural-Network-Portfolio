from setuptools import setup, find_packages

setup(
    name="neural-network-projects",
    version="1.0.0",
    description="A collection of neural network projects for portfolio showcase",
    author="Taimoor Khan",
    author_email="taimoor.khan@example.com",
    url="https://github.com/TaimoorKhan10/NeuralNetworkProjects",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.10.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "jupyter>=1.0.0",
        "pillow>=8.0.0",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 