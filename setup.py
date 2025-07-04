import json
from setuptools import setup, find_packages
import os

# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

# Function to read and parse the package.json file
def load_package_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Write the version to __init__.py
def write_version_to_init(version, init_file):
    with open(init_file, 'r') as file:
        lines = file.readlines()

    with open(init_file, 'w') as file:
        for line in lines:
            if line.startswith('__version__'):
                file.write(f'__version__ = "{version}"\n')
            else:
                file.write(line)

# Load package.json
package_info = load_package_json('package.json')

# Write version to __init__.py
init_file_path = os.path.join(package_info.get('name'), '__init__.py')
write_version_to_init(package_info.get('version'), init_file_path)

setup(
    name=package_info.get('name'),
    version=package_info.get('version'),
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    author=package_info.get('author'),
    author_email=package_info.get('author_email'),
    description=package_info.get('description'),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=package_info.get('url'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update with your actual license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)