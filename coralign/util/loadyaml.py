"""
Utility functions to load YAML files
"""

import yaml


def loadyaml(path, custom_exception=Exception):
    """
    Load a YAML file located at a given path

    Arguments:
     path: string containing path to file; can be absolute or relative

    Keyword arguments:
     custom_exception: Exception class to use when raising errors.  Defaults to
      Exception if none is specified.

    Returns:
     contents of YAML file as a Python object.  Note that this object does no
     parsing or validation of this data; this must be handled by the calling
     function.

    """

    # Load config from file
    try:
        with open(path) as f:
            raw_data = yaml.safe_load(f)
            pass
        pass
    # not a file
    except IOError:
        raise custom_exception('Config file does not exist.')
    # invalid YAML
    except yaml.YAMLError: # this is base class for all YAML errors
        raise custom_exception('File is not valid YAML.')
    except UnicodeDecodeError:
        raise custom_exception('File is not valid YAML.')

    return raw_data
