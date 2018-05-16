""" This module provides facilities to work with values of length in bits
that are not a multiple of 8.

"""

import struct

_int_formatter = struct.Struct('!I')


class BitsAccumulator:
    """ A tool to write bit values to a bytes array.

    This class allows the user to write values of a given number of bits, and outputs
    every time the total number written of bits exceeds a word.

    """
    def __init__(self):
        self.buffer = 0
        self.acc_bits = 0

    def push(self, value, num_bits) -> bytes:
        if num_bits > 32:
            raise ValueError('Can only push at most 32 bits at a time.')

        if num_bits + self.acc_bits <= 32:
            self.buffer = (self.buffer << num_bits) + value
            self.acc_bits += num_bits

            if self.acc_bits == 32:
                result = _int_formatter.pack(self.buffer)
                self.acc_bits = 0
                self.buffer = 0

                return result
            else:
                return b''
        else:
            num_leftover = num_bits + self.acc_bits - 32
            result = self.buffer << (32 - self.acc_bits)
            result += value >> num_leftover

            self.acc_bits = num_leftover
            self.buffer = value - ((value >> num_leftover) << num_leftover)

            return _int_formatter.pack(result)

    def flush(self):
        results = []

        while self.acc_bits > 8:
            shift = self.acc_bits - 8
            val = self.buffer >> shift
            results.append(val)
            self.buffer = self.buffer - (val << shift)
            self.acc_bits -= 8

        if self.acc_bits != 0:
            results.append(self.buffer << 8 - self.acc_bits)
            self.buffer = 0
            self.acc_bits = 0

        return bytes(results)


class BitsReader:
    """ A tool to read bit values from a given BytesIO.

    This class reads from the stream as needed, and outputs
    values of the given bits size as requested.

    """
    def __init__(self, stream):
        self.stream = stream
        self.buffer = 0
        self.buffer_bits = 0

    def read(self, num_bits):
        if self.buffer_bits >= num_bits:
            remaining_bits = self.buffer_bits - num_bits
            result = self.buffer >> remaining_bits
            self.buffer = self.buffer & ((1 << remaining_bits) - 1)
            self.buffer_bits = remaining_bits
            return result
        else:
            self.buffer = self.buffer << 8
            read_value = self.stream.read(1)
            if read_value == b'':
                raise ValueError('EOF when reading data.')

            self.buffer += struct.unpack('!B', read_value)[0]
            self.buffer_bits += 8

            return self.read(num_bits)


# A stream of bits that can be read. Because they come from an underlying byte stream,
# the total number of bits is always a multiple of 8. The bits are read in big endian.
class BitInputStream:
    """ A class to provide bit-level access to io streams. """
    def __init__(self, stream):
        """ Initialize a new bit input stream reading from the specified stream.

        Parameters
        ----------
        stream: The stream from which the input is read.
        """
        # The underlying byte stream to read from
        self._stream = stream
        # Either in the range [0x00, 0xFF] if bits are available, or -1 if end of stream is reached
        self.current_byte = 0
        # Number of remaining bits in the current byte, always between 0 and 7 (inclusive)
        self.num_bits_remaining = 0

    def read(self):
        """ Read a single bit from the stream.

        Returns
        -------
        The current bit, or -1 if the end of stream is reached.
        """
        if self.current_byte == -1:
            return -1
        if self.num_bits_remaining == 0:
            temp = self._stream.read(1)
            if len(temp) == 0:
                self.current_byte = -1
                return -1
            self.current_byte = temp[0]
            self.num_bits_remaining = 8
        assert self.num_bits_remaining > 0
        self.num_bits_remaining -= 1
        return (self.current_byte >> self.num_bits_remaining) & 1

    def read_eof(self):
        """ Read a bit from the stream.

        Returns
        -------
        The current bit.

        Raises
        ------
        EOFError: the underlying stream reached EOF.
        """
        result = self.read()
        if result != -1:
            return result
        else:
            raise EOFError()

    # Closes this stream and the underlying input stream.
    def close(self):
        self._stream.close()
        self.current_byte = -1
        self.num_bits_remaining = 0


# A stream where bits can be written to. Because they are written to an underlying
# byte stream, the end of the stream is padded with 0's up to a multiple of 8 bits.
# The bits are written in big endian.
class BitOutputStream(object):
    def __init__(self, stream):
        self._stream = stream  # The underlying byte stream to write to
        self.current_byte = 0  # The accumulated bits for the current byte, always in the range [0x00, 0xFF]
        self.num_bits_filled = 0  # Number of accumulated bits in the current byte, always between 0 and 7 (inclusive)

    def write(self, b):
        """ Writes a given bit to the stream.

        Parameters
        ----------
        b: 0, or 1, the bit to write.
        """
        if b not in (0, 1):
            raise ValueError("Argument must be 0 or 1")
        self.current_byte = (self.current_byte << 1) | b
        self.num_bits_filled += 1
        if self.num_bits_filled == 8:
            towrite = bytes((self.current_byte,))
            self._stream.write(towrite)
            self.current_byte = 0
            self.num_bits_filled = 0

    def close(self):
        """ Flushes the remaining unwritten bits. """
        while self.num_bits_filled != 0:
            self.write(0)
