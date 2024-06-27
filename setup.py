from setuptools import setup, find_packages

with open("README.md", "r") as rf:
    long_description = rf.read()
    
    
setup(
    name="ML&DL on Mac",
    version="0.1",
    author="Albert HHT",
    author_email="105294839+AlbertHHT@users.noreply.github.com",
    description="ML and DL on Mac ARM M2 structure, with limited hardware capacity",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/AlbertHHT/AI_MacOS.git",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "pandas", "matplotlib","tensorflow", "mlx", "transformers", "accelerate", "jax"],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS"
    ],
    python_requires=">=3.12",
)