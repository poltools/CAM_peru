from setuptools import setup, find_packages

setup(
    name="cam_peru",
    version="0.1.0",
    description=(
        "NLP pipeline for classifying, clustering, and network-analysing "
        "open-ended justifications for complementary and alternative medicine "
        "(CAM) use in Peru."
    ),
    python_requires=">=3.10,<3.13",
    packages=find_packages(include=["cam_peru", "cam_peru.*"]),
    include_package_data=True,
)
