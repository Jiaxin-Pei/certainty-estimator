import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="certainty-estimator",
    version="1.4",
    author="Jiaxin Pei",
    author_email="pedropei@umich.edu",
    description="This package is used to estimate certainty for scientific findings",
    python_requires=">=3.6",
    include_package_data=True,
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown"
    #package_data = {'':['data']}
)
