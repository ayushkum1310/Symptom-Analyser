from setuptools import setup,find_packages
from pathlib import Path
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


h='-e .'
def get_req(file_path:Path)->list:
    r=[]
    with open(file_path) as f:
        r=f.readlines()
        req=[re.replace("\n","") for re in r]
    if h in req:
        req.remove(h)
    return req

    



setup(
name='Disease Prediction',
version='0.0.1',
author='Ayush',
author_email='ayushpripl@gmail.com',
packages=find_packages(),
install_requires=get_req('requirements.txt')
)