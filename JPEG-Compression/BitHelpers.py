class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.buffer = 0
        self.nbits = 0

    def read_bit(self):
        if self.nbits == 0:
            self._fill_buffer()
        self.nbits -= 1
        return (self.buffer >> self.nbits) & 1

    def read_bits(self, n):
        value = 0
        for _ in range(n):
            value = (value << 1) | self.read_bit()
        return value

    def _fill_buffer(self):
        byte = self.data[self.pos]
        self.pos += 1
        if byte == 0xFF and self.pos < len(self.data) and self.data[self.pos] == 0x00:
            self.pos += 1
        self.buffer = byte
        self.nbits = 8


class BitWriter:
    def __init__(self):
        self.buffer = 0
        self.nbits = 0
        self.bytes = bytearray()

    def write_bit(self, bit):
        self.buffer = (self.buffer << 1) | (bit & 1)
        self.nbits += 1
        if self.nbits == 8:
            self._flush_byte()

    def write_bits(self, value, n):
        for i in range(n - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def _flush_byte(self):
        byte = self.buffer & 0xFF
        self.bytes.append(byte)
        if byte == 0xFF:
            self.bytes.append(0x00)
        self.buffer = 0
        self.nbits = 0

    def flush(self):
        if self.nbits > 0:
            self.buffer <<= (8 - self.nbits)
            self._flush_byte()

    def get_bytes(self):
        self.flush()
        return bytes(self.bytes)