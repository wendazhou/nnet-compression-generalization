""" This file implements the final part of a compression algorithm which
would correspond to writing a file in the encoded format.

The encoding format simply encodes the sparse weights, the codebook
and the value of the non-zero elements. It may then optionally apply
some general purpose compression algorithm to further reduce the size
of the input.

"""

import tensorflow as tf
import numpy as np
import io
import struct
import lzma
from ._bit_manipulation import BitsAccumulator, BitsReader


def _iter_max_size(values, max_value):
    for v in values:
        while v >= max_value:
            yield max_value - 1
            v -= max_value
        yield v


def get_bits_for_index(idx_diff):
    """ Get the number of bits to use to encode the index.

    Currently this uses a simplistic algorithm which attempts to encode the
    90th percentile of differences in the index.

    Parameters
    ----------
    idx_diff: The list of index differences to encode.

    Returns
    -------
    An integer between 0 and 16 representing the number of bits to use.
    """
    percentile_max = np.percentile(idx_diff, 90, interpolation='higher')

    if percentile_max == 0:
        return 0

    return min(int(np.ceil(np.log2(np.percentile(idx_diff, 90, interpolation='higher')))), 16)


def get_index_list(idx_diff, idx_diff_bits=None):
    """ Get the list of index differences.

    This function splits the values of idx_diff to ensure that all can be encoded
    in the given number of bits. In the compression, we split the gaps, and when
    one is too long we simply fill in the value with zeros as necessary.

    Parameters
    ----------
    idx_diff: The list of index differences to encode.
    idx_diff_bits: The number of bits used to encode the index.

    Returns
    -------
    A list of indices such that all can be encoded in the given number of bits.
    """
    if idx_diff_bits is None:
        idx_diff_bits = get_bits_for_index(idx_diff)

    return list(_iter_max_size(idx_diff, 2 ** idx_diff_bits))


def _compress_default(stream, values):
    """ Default compression algorithm which uses a fixed number of bits.

    Parameters
    ----------
    stream: The OutputWriter into which to write the encoded values.
    values: The values to encode.
    """
    ba = BitsAccumulator()
    num_bits = int(np.ceil(np.log2(np.max(values) + 1)))
    num_values = len(values)

    stream.write(ba.push(num_bits, num_bits=8))
    stream.write(ba.push(num_values, num_bits=32))

    for v in values:
        stream.write(ba.push(v, num_bits))

    stream.write(ba.flush())


def _decompress_default(stream):
    """ Default decompression algorithm which uses

    Parameters
    ----------
    stream: The stream from which to decompress.

    Returns
    -------
    A list of decoded values.
    """
    ba = BitsReader(stream)
    num_bits = ba.read(8)
    num_values = ba.read(32)

    data = np.empty(num_values, dtype=np.uint32)

    for i in range(num_values):
        data[i] = ba.read(num_bits)

    return data


class LzmaCompression:
    """ Compression strategy using standard LZMA.

    This class applies the standard LZMA compression to the sequence
    of bytes obtained by the given strategy.
    """
    _filters = [{
        'id': lzma.FILTER_LZMA2,
        'preset': 9
    }]

    def __init__(self, compression=None):
        if compression is None:
            self._compress = _compress_default
            self._decompress = _decompress_default
        else:
            self._compress = compression.compress
            self._decompress = compression.decompress

    def compress(self, stream, values):
        buffer = io.BytesIO()
        self._compress(buffer, values)
        compressed = lzma.compress(
            buffer.getvalue(), format=lzma.FORMAT_RAW, filters=LzmaCompression._filters)
        stream.write(struct.pack('!I', len(compressed)))
        stream.write(compressed)

    def decompress(self, stream):
        length = struct.unpack('!I', stream.read(4))[0]
        data = stream.read(length)
        decompressed = lzma.decompress(data, format=lzma.FORMAT_RAW, filters=LzmaCompression._filters)
        return self._decompress(io.BytesIO(decompressed))


class EntropyEstimateCompression:
    """ A compression strategy that does not actually compress,
    but instead reports the entropy of the values it is given.

    This gives a lower bound on the compressed size of the data.
    """
    def compress(self, stream, values):
        _, counts = np.unique(values, return_counts=True)
        length_bits = -np.sum(counts * np.log2(counts / np.sum(counts)))
        length_bytes = int(np.ceil(length_bits / 8))
        stream.write(b'0' * length_bytes)

    def decompress(self, stream):
        raise NotImplemented('This strategy does not implement decompression.')


def _get_compression(compressor):
    if compressor is None:
        return _compress_default
    else:
        return compressor.compress


def compress_variable(value, output=None,
                      codebook_dtype=np.float16,
                      compression_index=None,
                      compression_weights=None):
    """ This function compresses the given variable into
    a compressed representation storing the codebook of non-zero
    values, indexes of non-zero values and quantized values.

    This does not store the shape of the variable, and must be
    passed in again to be restored.

    The format is given as follows:

    - byte 1: the upper 4 bits encode the number of bits used for
        the quantized value, the lower 4 bits encode the number of
        bits used for the offset.
    - short: a short representing the length of the codebook
        excluding the zero value (each codebook value is represented
        as a single precision floating point number).
    - int: an integer representing the number of non-zero elements
        stored in the tensor.
    - codebook: a sequence of floats in IEE-754 single precision format
        corresponding to the codebook in order.
    - values: a sequence of pairs of values of given number of bits
        in byte 1 representing the offset - 1 and the quantized value.

    The number of bytes written is rounded to the nearest byte of the
    total code length.

    Parameters
    ----------
    value: a numpy array containing the values of the variable
        to store. These must be already quantized values.
    output: a BytesIO to which to write the compressed representation.
    codebook_dtype: a numpy type to indicate the data type to use
        to encode codebook values.
    compression_index: Whether to use any additional compressor to encode
        the indices.
    compression_weights: Whether to use any additional compressor to encode
        the quantized values.

    Returns
    -------
    bytes: the representation of the variable in compressed format.
    """
    if output is None:
        output = io.BytesIO()

    value = np.ravel(value)
    unique_values = np.unique(value)

    zero_in_values = False
    codebook = {0.0: 0}
    codebook_values = []
    code = 1

    for v in unique_values:
        if v != 0:
            codebook[v] = code
            codebook_values.append(v)
            code += 1
        else:
            zero_in_values = True

    if len(codebook) > 2 ** 16:
        raise ValueError('Too many distinct values in variable!')

    idx = np.flatnonzero(value)

    if len(idx) == 0:
        output.write(struct.pack('BB', 0, 0))
        return output

    idx_diff_min_one = np.ediff1d(idx, to_begin=idx[0] + 1) - 1

    # Determine number of bits to use for index difference.
    num_bits_idx = get_bits_for_index(idx_diff_min_one)

    if num_bits_idx == 0 and not zero_in_values:
        # We are storing a dense matrix.
        codebook.pop(0)
        for k in codebook.keys():
            codebook[k] = codebook[k] - 1

    # Build the actual list of index differences such that they can all
    # be represented in the adequate number of bits.
    idx_diff_list = get_index_list(idx_diff_min_one, idx_diff_bits=num_bits_idx)

    # Encode header information
    output.write(struct.pack('!H', len(codebook) - 1))

    # Encode codebook
    for code_value in codebook_values:
        output.write(np.array(code_value, dtype=codebook_dtype).tobytes())

    compression_index = _get_compression(compression_index)
    compression_weights = _get_compression(compression_weights)

    # Encode index diff list
    if num_bits_idx != 0:
        compression_index(output, idx_diff_list)

    code_values = np.zeros(len(idx_diff_list), dtype=np.uint32)
    current_idx = -1

    for i, d in enumerate(idx_diff_list):
        current_idx += d + 1
        v = value[current_idx]
        code_values[i] = codebook[v]

    compression_weights(output, code_values)
    return output


def decompress_variable(code, shape, codebook_dtype=np.float16, compression=None):
    """ Decompress a variable.

    This function is the inverse of the `compress_variable`
    function.

    To perform the decompression, the original shape of the variable must
    be known. In neural networks, this is a property of the model and
    is thus not encoded in the code.

    Parameters
    ----------
    code: The compressed code representing the variable.
    shape: The shape of the variable to decode.
    codebook_dtype: The type of the floating point numbers in the codebook.
    compression: The type of compression to use.

    Returns
    -------
    data: a numpy array of the given shape representing the decoded
        variable information.
    """
    if hasattr(code, 'read'):
        data = code
    else:
        data = io.BytesIO(code)

    result = np.zeros(np.prod(shape), dtype=np.float32)

    br = BitsReader(data)

    codebook_len = br.read(16)
    codebook = {0: 0.0}

    for i in range(codebook_len):
        if codebook_dtype is np.float16:
            raw = br.read(16).to_bytes(2, byteorder='big')
        elif codebook_dtype is np.float32:
            raw = br.read(32).to_bytes(4, byteorder='big')
        else:
            raise ValueError('Invalid codebook data type')

        codebook[i + 1] = np.frombuffer(raw, dtype=codebook_dtype, count=1)[0]

    if compression is None:
        decompress = _decompress_default
    else:
        decompress = compression.decompress

    idx_diff_list = decompress(data)
    values = decompress(data)

    current_index = -1
    for c, d in zip(values, idx_diff_list):
        current_index += d + 1
        v = codebook[c]

        result[current_index] = v

    return np.reshape(result, shape)


def compress_checkpoint(checkpoint, variables, compression=None) -> bytes:
    """ Obtains a compressed representation of the given variables in the checkpoint.
    This function assumes that the weights in the checkpoints have already been pruned
    and quantized, and then encodes the checkpoint into an codebook + index + compressed
    values.

    Parameters
    ----------
    checkpoint: the checkpoint to load the variables from.
    variables: the variables to compress.
    compression: A compression strategy to use for the codebooks.

    Returns
    -------
    A byte string representing the compressed representation of the tensor.
    """
    if isinstance(checkpoint, str):
        checkpoint = tf.train.load_checkpoint(checkpoint)

    output = io.BytesIO()

    for variable_name in variables:
        variable_value = checkpoint.get_tensor(variable_name)
        compress_variable(variable_value, output, compression_index=compression)

    data = output.getvalue()

    return data
