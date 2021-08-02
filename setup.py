from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto ensemble", # 이 패키지를 설치/삭제할 때 사용할 이름을 의미한다. 이 이름과 import할 때 쓰이는 이름은 다르다.
    version="1.0.0",
    author="Hosu Lee",
    author_email="leehosu01@naver.com",
    description="The simplest ensemble library machine learning model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leehosu01/ensemble",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
)
