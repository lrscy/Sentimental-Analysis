import copy
import torch.nn as nn


def clones(module, no_of_copies):
    """Produce no_of_copies identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(no_of_copies)])


def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8')
    else:
        raise ValueError('Unsupported string type: %s' % (type(text)))

