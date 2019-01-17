import setuptools

setuptools.setup(
    name='audionmf',
    author="Tomáš Drbota",
    author_email="argoneuscze@gmail.com",
    description="A tool used for compressing audio using NMF.",

    long_description=open('README.md').read(),
    version='0.0.1',
    packages=setuptools.find_packages(),
    license='MIT',

    install_requires=[
        "click",
        "numpy",
        "scipy>=1.2.0",
        'nimfa'
    ],

    setup_requires=["pytest-runner"],

    tests_require=["pytest"],

    entry_points={
        'console_scripts': [
            'audionmf = audionmf.__main__:main',
        ],
    }
)
