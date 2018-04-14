"""This module implements support for various weight quantization strategies."""

from ._getter import apply_quantization, quantized_variable_getter, assign_quantized_values_to_weights, \
    QUANTIZED_VARIABLE_COLLECTION, CommitQuantizedValueHook


from ._minmax import minmax_quantization

from ._codebook import codebook_quantization
