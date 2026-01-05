import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from scipy.fft import dctn, idctn
from BitHelpers import BitWriter, BitReader
from JPEGTables import JPEGTables

class JPEGCompressor:
      BLOCK_SIZE = 8
      Q_LUMINANCE = JPEGTables.Q_LUMINANCE
      Q_CHROMINANCE = JPEGTables.Q_CHROMINANCE
      BITS_DC_LUMINANCE = JPEGTables.BITS_DC_LUMINANCE
      HUFFVAL_DC_LUMINANCE = JPEGTables.HUFFVAL_DC_LUMINANCE
      BITS_DC_CHROMINANCE = JPEGTables.BITS_DC_CHROMINANCE
      HUFFVAL_DC_CHROMINANCE = JPEGTables.HUFFVAL_DC_CHROMINANCE
      BITS_AC_LUMINANCE = JPEGTables.BITS_AC_LUMINANCE
      HUFFVAL_AC_LUMINANCE = JPEGTables.HUFFVAL_AC_LUMINANCE
      BITS_AC_CHROMINANCE = JPEGTables.BITS_AC_CHROMINANCE
      HUFFVAL_AC_CHROMINANCE = JPEGTables.HUFFVAL_AC_CHROMINANCE
      ZIGZAG_ORDER = JPEGTables.ZIGZAG_ORDER
      
      def __init__(self, quality=50):
            self.quality = quality
            self.dc_lum_table = self._build_huffman_table(self.BITS_DC_LUMINANCE, self.HUFFVAL_DC_LUMINANCE)
            self.dc_chrom_table = self._build_huffman_table(self.BITS_DC_CHROMINANCE, self.HUFFVAL_DC_CHROMINANCE)
            self.ac_lum_table = self._build_huffman_table(self.BITS_AC_LUMINANCE, self.HUFFVAL_AC_LUMINANCE)
            self.ac_chrom_table = self._build_huffman_table(self.BITS_AC_CHROMINANCE, self.HUFFVAL_AC_CHROMINANCE)
            self.dc_lum_decode_table = self._build_reverse_huffman_table(self.BITS_DC_LUMINANCE, self.HUFFVAL_DC_LUMINANCE)
            self.dc_chrom_decode_table = self._build_reverse_huffman_table(self.BITS_DC_CHROMINANCE, self.HUFFVAL_DC_CHROMINANCE)
            self.ac_lum_decode_table = self._build_reverse_huffman_table(self.BITS_AC_LUMINANCE, self.HUFFVAL_AC_LUMINANCE)
            self.ac_chrom_decode_table = self._build_reverse_huffman_table(self.BITS_AC_CHROMINANCE, self.HUFFVAL_AC_CHROMINANCE)
            
      def _build_huffman_table(self, bits, huffval):
            huffman_table = {}
            code = 0
            k = 0
            for i in range(1, len(bits)):
                  for _ in range(bits[i]):
                        huffman_table[huffval[k]] = (code, i)
                        code += 1
                        k += 1
                  code <<= 1
            return huffman_table
      
      def _build_reverse_huffman_table(self, bits, huffval):
            decode_table = {}
            code = 0
            k = 0
            for i in range(1, len(bits)):
                  for _ in range(bits[i]):
                        decode_table[(code, i)] = huffval[k]
                        code += 1
                        k += 1
                  code <<= 1
            return decode_table
      
      def _quality_to_scale(self):
            quality = np.clip(self.quality, 1, 100)
            if quality < 50:
                  return 50 / quality
            else:
                  return (100 - quality) / 50
            
      def _rgb_to_ycbcr(self, rgb_img):
            rgb_img = rgb_img.astype(np.float64)  
            transform_matrix = np.array([
            [ 0.299,      0.587,      0.114    ],  # Y
            [-0.168736,  -0.331264,   0.5      ],  # Cb
            [ 0.5,       -0.418688,  -0.081312 ]   # Cr
            ])
            ycbcr = np.dot(rgb_img, transform_matrix.T)
            ycbcr[:, :, 1:] += 128
            return ycbcr

      def _ycbcr_to_rgb(self, ycbcr_img):
            ycbcr_img = ycbcr_img.astype(np.float64)
            ycbcr_img[:, :, 1:] -= 128
            inverse_transform_matrix = np.array([
            [1.0,  0.0,       1.402    ],  # R
            [1.0, -0.344136, -0.714136 ],  # G
            [1.0,  1.772,     0.0      ]   # B
            ])
            rgb = np.dot(ycbcr_img, inverse_transform_matrix.T)
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            return rgb
      
      def _is_gray(self, img):
            if len(img.shape) == 2:
                  return True
            if len(img.shape) == 3 and img.shape[2] == 1:
                  return True
            return False
      
      def _pad(self, img):
            if img.ndim == 2:
                  h, w = img.shape
                  pad_h = (self.BLOCK_SIZE - h % self.BLOCK_SIZE) % self.BLOCK_SIZE
                  pad_w = (self.BLOCK_SIZE - w % self.BLOCK_SIZE) % self.BLOCK_SIZE
                  return np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
            else:
                  h, w, _ = img.shape
                  pad_h = (self.BLOCK_SIZE - h % self.BLOCK_SIZE) % self.BLOCK_SIZE
                  pad_w = (self.BLOCK_SIZE - w % self.BLOCK_SIZE) % self.BLOCK_SIZE
                  return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
      
      def _quantize_block(self, q_matrix, block):
            scale = self._quality_to_scale()
            q_scaled = np.maximum(q_matrix * scale, 1)
            x = block.astype(np.float64) - 128
            y = dctn(x, norm='ortho')
            coeffs = np.round(y / q_scaled).astype(np.int32)
            return coeffs

      def _dequantize_block(self, q_matrix, coeffs):
            scale = self._quality_to_scale()
            q_scaled = np.maximum(q_matrix * scale, 1)
            y_jpeg = coeffs * q_scaled
            x_jpeg = idctn(y_jpeg, norm='ortho') + 128
            return np.clip(x_jpeg, 0, 255).astype(np.uint8)
      
      def _zigzag(self, block):
            return np.array([block[i,j] for i,j in self.ZIGZAG_ORDER])
      
      def _zigzag_reverse(self, arr):
            block = np.zeros((8, 8), dtype=arr.dtype)
            for index, (i,j) in enumerate(self.ZIGZAG_ORDER):
                  block[i,j] = arr[index]
            return block
      
      def _get_size(self, value):
            if value == 0:
                  return 0
            return int(np.floor(np.log2(abs(value))) + 1)
      
      def _get_additional_bits(self, value, size):
            if value > 0:
                  return value
            else:
                  return (1 << size) + value - 1
      
      def _encode_dc(self, dc, prev_dc=0):
            dc_diff = dc - prev_dc
            size = self._get_size(dc_diff)
            if size == 0:
                  additional_bits = 0
            else:
                  additional_bits = self._get_additional_bits(dc_diff, size)
            return size, additional_bits

      def _decode_dc(self, size, additional_bits, prev_dc=0):
            if size == 0:
                  dc_diff = 0
            else:
                  if additional_bits >= (1 << (size - 1)):
                        dc_diff = additional_bits
                  else:
                        dc_diff = additional_bits - (1 << size) + 1
            dc = prev_dc + dc_diff
            return dc
      
      def _encode_ac(self, ac_coeffs):
            symbols = []
            run_length = 0
            for coeff in ac_coeffs:
                  if coeff == 0:
                        run_length += 1
                  else:
                        while run_length > 15:
                              symbols.append((15, 0, 0))
                              run_length -= 16
                        size = self._get_size(coeff)
                        additional_bits = self._get_additional_bits(coeff, size)
                        symbols.append((run_length, size, additional_bits))
                        run_length = 0
            if run_length > 0:
                  symbols.append((0, 0, 0))
            return symbols
      
      def _decode_ac(self, symbols):
            ac_coeffs = []
            for run_length, size, additional_bits in symbols:
                  if run_length == 0 and size == 0:
                        ac_coeffs.extend([0] * (63 - len(ac_coeffs)))
                        break
                  ac_coeffs.extend([0] * run_length)
                  if size == 0:
                        coeff = 0
                  else:
                        if additional_bits >= (1 << (size - 1)):
                              coeff = additional_bits
                        else:
                              coeff = additional_bits - (1 << size) + 1
                  ac_coeffs.append(coeff)
            return ac_coeffs[:64]
      
      def _read_huffman_symbol(self, reader, decode_table):
            code = 0
            for length in range(1, 17):
                  bit = reader.read_bit()
                  code = (code << 1) | bit
                  if (code, length) in decode_table:
                        return decode_table[(code, length)]
            raise ValueError("Invalid Huffman code!")

      def _compress_channel(self, q_matrix, channel, type='luminance'):
            h, w = channel.shape
            if type == 'luminance':
                  dc_table = self.dc_lum_table
                  ac_table = self.ac_lum_table
            elif type == 'chrominance':
                  dc_table = self.dc_chrom_table
                  ac_table = self.ac_chrom_table
            else:
                  raise ValueError("Invalid channel type!")
            
            writer = BitWriter()
            prev_dc = 0
            block_count = 0
            
            for i in range(0, h, self.BLOCK_SIZE):
                  for j in range(0, w, self.BLOCK_SIZE):
                        block = channel[i:i+self.BLOCK_SIZE, j:j+self.BLOCK_SIZE]
                        coeffs = self._quantize_block(q_matrix, block)
                        zigzag = self._zigzag(coeffs)

                        dc_size, dc_bits = self._encode_dc(zigzag[0], prev_dc)
                        code, code_len = dc_table[dc_size]
                        writer.write_bits(code, code_len)
                        if dc_size > 0:
                              writer.write_bits(dc_bits, dc_size)
                        prev_dc = zigzag[0]
                        
                        ac_symbols = self._encode_ac(zigzag[1:])
                        for run_length, size, additional_bits in ac_symbols:
                              symbol = (run_length << 4) | size
                              code, code_len = ac_table[symbol]
                              writer.write_bits(code, code_len)
                              if size > 0:
                                    writer.write_bits(additional_bits, size)
                        block_count += 1
            
            writer.flush()
            return writer.get_bytes(), block_count
      
      def _decompress_channel(self, q_matrix, data, shape, block_count, type='luminance'):
            h, w = shape
            channel = np.zeros((h, w), dtype=np.uint8)
            if type == 'luminance':
                  dc_decode_table = self.dc_lum_decode_table
                  ac_decode_table = self.ac_lum_decode_table
            elif type == 'chrominance':
                  dc_decode_table = self.dc_chrom_decode_table
                  ac_decode_table = self.ac_chrom_decode_table
            else:
                  raise ValueError("Invalid channel type!")
            
            reader = BitReader(data)
            prev_dc = 0
            block_idx = 0
            
            for i in range(0, h, self.BLOCK_SIZE):
                  for j in range(0, w, self.BLOCK_SIZE):
                        if block_idx >= block_count:
                              break
                        dc_size = self._read_huffman_symbol(reader, dc_decode_table)
                        if dc_size == 0:
                              dc_bits = 0
                        else:
                              dc_bits = reader.read_bits(dc_size)
                        dc_coeff = self._decode_dc(dc_size, dc_bits, prev_dc)
                        prev_dc = dc_coeff
                        
                        ac_symbols = []
                        ac_count = 0
                        while ac_count < 63:
                              symbol = self._read_huffman_symbol(reader, ac_decode_table)
                              run_length = (symbol >> 4) & 0x0F
                              size = symbol & 0x0F
                              if run_length == 0 and size == 0:
                                    ac_symbols.append((0, 0, 0))
                                    break
                              additional_bits = 0
                              if size > 0:
                                    additional_bits = reader.read_bits(size)
                              ac_symbols.append((run_length, size, additional_bits))
                              ac_count += run_length + 1
                        ac_coeffs = self._decode_ac(ac_symbols)
                        
                        zigzag = np.zeros(64, dtype=np.int32)
                        zigzag[0] = dc_coeff
                        zigzag[1:] = ac_coeffs[:63]
                        coeffs = self._zigzag_reverse(zigzag)
                        block = self._dequantize_block(q_matrix, coeffs)
                        channel[i:i+self.BLOCK_SIZE, j:j+self.BLOCK_SIZE] = block
                        
                        block_idx += 1
            return channel
      
      def compress(self, img):
            original_shape = img.shape
            if self._is_gray(img):
                  img_padded = self._pad(img)
                  compressed_data, num_blocks = self._compress_channel(self.Q_LUMINANCE, img_padded, type='luminance')
                  return {
                        'data': compressed_data,
                        'shape': img_padded.shape,
                        'original_shape': original_shape,
                        'num_blocks': num_blocks,
                        'is_color': False,
                        'quality': self.quality
                  }
            else:
                  ycbcr = self._rgb_to_ycbcr(img)
                  ycbcr_padded = self._pad(ycbcr)
                  y_data, y_blocks = self._compress_channel(self.Q_LUMINANCE, ycbcr_padded[:, :, 0], type='luminance')
                  cb_data, cb_blocks = self._compress_channel(self.Q_CHROMINANCE, ycbcr_padded[:, :, 1], type='chrominance')
                  cr_data, cr_blocks = self._compress_channel(self.Q_CHROMINANCE, ycbcr_padded[:, :, 2], type='chrominance')
                  return {
                        'y_data': y_data, 'cb_data': cb_data, 'cr_data': cr_data,
                        'shape': ycbcr_padded.shape,
                        'original_shape': original_shape,
                        'num_blocks': (y_blocks, cb_blocks, cr_blocks),
                        'is_color': True,
                        'quality': self.quality
                  }
                  
      def decompress(self, compressed_data):
            self.quality = compressed_data['quality']
            if not compressed_data['is_color']:
                  channel = self._decompress_channel(self.Q_LUMINANCE, compressed_data['data'], compressed_data['shape'], compressed_data['num_blocks'],type='luminance')
                  original_h, original_w = compressed_data['original_shape']
                  return channel[:original_h, :original_w]
            else:
                  h, w, _ = compressed_data['shape']
                  y = self._decompress_channel(self.Q_LUMINANCE, compressed_data['y_data'], (h, w), compressed_data['num_blocks'][0], type='luminance')
                  cb = self._decompress_channel(self.Q_CHROMINANCE, compressed_data['cb_data'], (h, w), compressed_data['num_blocks'][1], type='chrominance')
                  cr = self._decompress_channel(self.Q_CHROMINANCE, compressed_data['cr_data'], (h, w), compressed_data['num_blocks'][2], type='chrominance')
                  
                  ycbcr = np.stack((y, cb, cr), axis=-1)
                  rgb = self._ycbcr_to_rgb(ycbcr)
                  original_h, original_w, _ = compressed_data['original_shape']
                  return rgb[:original_h, :original_w, :]
      
      def set_quality(self, quality):
            self.quality = quality


def get_mse(img1, img2):
      return np.mean((img1 - img2) ** 2)

if __name__ == "__main__":
      x = datasets.face()

      fig, ax = plt.subplots(1, 2, figsize=(15, 5))
      ax[0].imshow(x)
      ax[0].set_title("Original")

      target_mse = 50
      q_min, q_max = 1, 100
      compressor = JPEGCompressor()

      for _ in range(25):
            q_mid = (q_min + q_max) // 2
            compressor.set_quality(q_mid)
            compressed = compressor.compress(x)
            x_jpeg = compressor.decompress(compressed)

            mse = get_mse(x.astype(np.float64), x_jpeg.astype(np.float64))
            if mse < target_mse:
                  q_max = q_mid
            else:
                  q_min = q_mid

            if q_max - q_min <= 1:
                  break
            
      original_size = x.nbytes 
      compressed_size = len(compressed['y_data']) + len(compressed['cb_data']) + len(compressed['cr_data'])
      ratio = (1 - compressed_size / original_size) * 100
      print(f"JPEG Compression: {ratio:.1f}% reduction ({original_size//1024}KB → {compressed_size//1024}KB)")
      ax[1].imshow(x_jpeg)
      ax[1].set_title(f"JPEG (Q={q_mid}, MSE={mse:.2f})")
      plt.tight_layout()
      plt.show()
      plt.imsave("image_uncompressed.png", x)
      plt.imsave("image_compressed.png", x_jpeg)