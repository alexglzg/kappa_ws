import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'kappa_experiments'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alex',
    maintainer_email='alex_gg97@hotmail.com',
    description='Predefined corridor scenarios for standalone tests of the kappa analytical planner.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'experiment_node = kappa_experiments.experiment_node:main',
            'metrics_node = kappa_experiments.metrics_node:main',
            'postprocess = kappa_experiments.postprocess:main',
        ],
    },
)
