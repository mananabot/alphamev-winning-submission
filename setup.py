from setuptools import setup, find_packages


setup(
    name='mev-competition',
    author='David Hlinstak',
    author_email="d.hlinstak@gmail.com",
    description='MEV competition submission',
    version="0.1.9",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'web3',
        'python-dotenv',
        'gensim',
        'dnspython',
        'azureml-sdk',
        'azureml-core',
        'pyngrok',
        'mljar-supervised'
    ]
)
