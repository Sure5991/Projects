from setuptools import find_packages,setup
from typing import List

hypen_e_dot = '-e .'

# Requirement List
def get_requirement(file_path:str)->List[str]:

    '''
    This will return list of requirements
    '''
    requirement = []
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        requirement = [req.replace('\n','') for req in requirement]
        if hypen_e_dot in requirement:
            requirement.remove(hypen_e_dot)
    return requirement

# Setup File
setup(
    name = "end-to-to-ml-project",
    version= '0.1',
    author = 'Suresh R',
    author_email= 'rsuresh5991@gmail.com',
    packages = find_packages(),
    install_requires= get_requirement('requirements.txt')
)