import setuptools

setuptools.setup(
    name='audionmf',
    author="Tomáš Drbota",
    author_email="argoneuscze@gmail.com",
    description="A tool used for compressing audio using NMF.",

    long_description=open('README').read(),
    version='0.0.1',
    packages=setuptools.find_packages(),
    license='MIT',

    entry_points={
        'console_scripts': [
            'audionmf = audionmf.main:main',
        ],
    }
)
