from setuptools import setup, find_packages

setup(
    name="rag_showdown",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "requests",
        "python-dotenv",
        "tqdm",
        "pydantic",
    ],
)
