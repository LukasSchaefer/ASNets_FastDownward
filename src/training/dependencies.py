from argparse import Namespace

# Configure which dependencies are available in your environment.
# Some modules are only loaded if their dependencies are met.

DEPENDENCIES = Namespace(
    keras=True,
    tensorflow=True
)