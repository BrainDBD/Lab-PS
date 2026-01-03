import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from scipy.fft import dctn, idctn

class JPEGCompressor:
      Q_LUMINANCE = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 28, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
            ], dtype=np.float64)
      Q_CHROMINANCE = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
            ], dtype=np.float64)
      BLOCK_SIZE = 8
      
      def __init__(self, quality=50):
            self.quality = quality
            
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
            y_jpeg = np.round(y / q_scaled) * q_scaled
            x_jpeg = idctn(y_jpeg, norm='ortho') + 128
            
            return np.clip(x_jpeg, 0, 255)
      
      def _compress_channel(self, q_matrix, channel):
            h, w = channel.shape
            compressed = np.zeros_like(channel, dtype=np.float64)
            for i in range(0, h, self.BLOCK_SIZE):
                  for j in range(0, w, self.BLOCK_SIZE):
                        block = channel[i:i+self.BLOCK_SIZE, j:j+self.BLOCK_SIZE]
                        compressed[i:i+self.BLOCK_SIZE, j:j+self.BLOCK_SIZE] = self._quantize_block(q_matrix, block)
            return compressed
      
      def compress(self, img):
            original_shape = img.shape
            if self._is_gray(img):
                  img_padded = self._pad(img)
                  compressed = self._compress_channel(self.Q_LUMINANCE, img_padded)
                  return compressed[:original_shape[0], :original_shape[1]]
            else:
                  ycbcr = self._rgb_to_ycbcr(img)
                  ycbcr_padded = self._pad(ycbcr)
                  y_jpeg = self._compress_channel(self.Q_LUMINANCE, ycbcr_padded[:, :, 0])
                  cb_jpeg = self._compress_channel(self.Q_CHROMINANCE, ycbcr_padded[:, :, 1])
                  cr_jpeg = self._compress_channel(self.Q_CHROMINANCE, ycbcr_padded[:, :, 2])
                  compressed_ycbcr = np.stack((y_jpeg, cb_jpeg, cr_jpeg), axis=-1)
                  compressed_rgb = self._ycbcr_to_rgb(compressed_ycbcr)
                  return compressed_rgb[:original_shape[0], :original_shape[1], :]
      
      def set_quality(self, quality):
            self.quality = quality


def get_mse(img1, img2):
      return np.mean((img1 - img2) ** 2)

if __name__ == "__main__":
      x = datasets.face()
      fig, ax = plt.subplots(1, 2, figsize=(15, 5))
      ax[0].imshow(x)
      ax[0].set_title("Original Image")

      target_mse = 50
      q_min, q_max = 0, 100
      compressor = JPEGCompressor()
      for _ in range(30):
            q_mid = (q_min + q_max) // 2
            compressor.set_quality(q_mid)
            x_jpeg = compressor.compress(x)
            mse = get_mse(x, x_jpeg)
            
            if mse < target_mse:
                  q_max = q_mid
            else:
                  q_min = q_mid
            if q_max - q_min <= 1:
                  break
            
      ax[1].imshow(x_jpeg)
      ax[1].set_title(f"Compressed Image (Quality: {q_mid} | MSE: {mse:.2f})")
      plt.tight_layout()
      plt.savefig("JPEG-Compression/Compression_MSE.pdf")
      plt.show()
