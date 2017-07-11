from setuptools import setup
from setuptools import find_packages

required = ["pandas"]
if __name__ == '__main__':
  
  setup(name='trainer',
        install_requires = required ,
        packages=find_packages() ,
        include_package_data=True,
  )
