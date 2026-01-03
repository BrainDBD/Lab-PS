import cv2
from JPEGCompression import JPEGCompressor

class VideoCompressor:
    def __init__(self, quality=50):
        self.image_compressor = JPEGCompressor(quality)
    
    def compress_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file {input_path}!")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Cannot create video writer for {output_path}!")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            compressed_frame = self.image_compressor.compress(frame_rgb)
            compressed_bgr = cv2.cvtColor(compressed_frame.astype('uint8'), cv2.COLOR_RGB2BGR)
            out.write(compressed_bgr)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        out.release()
        print(f"Video compression complete! Video saved at {output_path}!")


if __name__ == "__main__":
    input_video = 'waiting.mp4'
    output_video = 'waiting_compressed.mp4'
    compressor = VideoCompressor()
    compressor.compress_video(input_video, output_video)