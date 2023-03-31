# Quality Checks Container
This Docker container provides a complete environment for running data quality checks using Great Expectations.

## Getting started
To use this container, you will need to have Docker installed on your machine. Once Docker is installed, you can build the container by running the following command:
```
docker build -t quality_checks .
```

This will build a Docker image called *quality_checks* that contains Great Expectations and all its dependencies.

### Running quality checks
To run data quality checks using Great Expectations in this container, you can use the Great Expectations Jupyter notebooks.