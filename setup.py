from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """
    This function will return list of requirements
    """
    requirement_list: List[str] = []

    with open("requirements.txt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirement_list.append(line)

    return requirement_list


setup(
    name="sensor",
    version="0.0.1",
    author="ineuron",
    author_email="avnish@ineuron.ai",
    packages=find_packages(),
    install_requires=get_requirements(),
)

