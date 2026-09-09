from setuptools import setup
import os
from glob import glob

package_name = 'box_sim'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Simple box robot simulation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'box_robot_simulator = box_sim.box_robot_simulator:main',
        ],
    },
)
