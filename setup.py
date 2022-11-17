from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'My F110 PhD results'
LONG_DESCRIPTION = 'f110 planning agents for safe autonomous racing'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="TrajectoryAidedLearning", 
        version=VERSION,
        author="Benjamin Evans",
        author_email="<bdevans@sun.ac.za>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'autonomous racing'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux",
        ]
)
