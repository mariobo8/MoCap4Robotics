import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class DotPattern:
    positions: List[Tuple[int, int]]

class MockCamera:
    def __init__(self, camera_ids: List[int], fps: List[int], resolution, colour: bool = True, config: str = "cube"):
        """
        Initialize mock camera with specified configuration.
        
        Args:
            camera_ids: List of camera IDs
            fps: List of frame rates for each camera
            resolution: Camera resolution ("large" or "small")
            colour: Whether to generate color frames
            config: Point configuration ("cube" or "plane")
        """
        self.camera_ids = camera_ids
        self.num_cameras = len(camera_ids)
        self._fps = fps
        self._colour = colour
        self.config = config
        
        # Set resolution
        if resolution == "large":
            self._width, self._height = 640, 480
        else:
            self._width, self._height = 320, 240
            
        # Camera settings
        self._exposure = [100] * self.num_cameras
        self._gain = [10] * self.num_cameras
        
        # Initialize patterns based on selected configuration
        self._patterns = self._get_patterns(config)

    def _get_patterns(self, config: str) -> List[DotPattern]:
        """Get dot patterns based on configuration."""
        if config == "cube":
            return self._get_cube_patterns()
        elif config == "plane":
            return self._get_plane_patterns()
        else:
            raise ValueError(f"Unknown configuration: {config}. Use 'cube' or 'plane'.")

    def _get_cube_patterns(self) -> List[DotPattern]:
        """Get patterns for cube configuration (current 8-point pattern)."""
        # Base pattern (for middle camera)
        base_pattern = [
            (220, 140),   # Top front left
            (420, 140),   # Top front right
            (220, 340),   # Bottom front left
            (420, 340),   # Bottom front right
            (260, 180),   # Top back left
            (380, 180),   # Top back right
            (260, 300),   # Bottom back left
            (380, 300),   # Bottom back right
        ]
        
        patterns = [
            # Camera 1 (left rotated view)
            [
                (180, 140),   # Top front left
                (380, 140),   # Top front right
                (180, 340),   # Bottom front left
                (380, 340),   # Bottom front right
                (200, 180),   # Top back left
                (340, 180),   # Top back right
                (200, 300),   # Bottom back left
                (340, 300),   # Bottom back right
            ],
            # Camera 2 (front view)
            base_pattern,
            # Camera 3 (right rotated view)
            [
                (260, 140),   # Top front left
                (460, 140),   # Top front right
                (260, 340),   # Bottom front left
                (460, 340),   # Bottom front right
                (300, 180),   # Top back left
                (440, 180),   # Top back right
                (300, 300),   # Bottom back left
                (440, 300),   # Bottom back right
            ]
        ]
        
        return [DotPattern(p) for p in patterns]

    def _get_plane_patterns(self) -> List[DotPattern]:
        """Get patterns for plane configuration (8 points in a flat plane)."""
        # Define a flat plane of points that will appear to rotate between cameras
        base_points = [
            (220, 120),   # Top left
            (320, 120),   # Top middle
            (420, 120),   # Top right
            (220, 240),   # Middle left
            (420, 240),   # Middle right
            (220, 360),   # Bottom left
            (320, 360),   # Bottom middle
            (420, 360),   # Bottom right
        ]

        patterns = [
            # Camera 1 (left rotated view - points compressed on right)
            [
                (160, 120),   # Top left
                (240, 120),   # Top middle
                (320, 120),   # Top right
                (160, 240),   # Middle left
                (320, 240),   # Middle right
                (160, 360),   # Bottom left
                (240, 360),   # Bottom middle
                (320, 360),   # Bottom right
            ],
            # Camera 2 (front view - evenly spaced)
            base_points,
            # Camera 3 (right rotated view - points compressed on left)
            [
                (320, 120),   # Top left
                (400, 120),   # Top middle
                (480, 120),   # Top right
                (320, 240),   # Middle left
                (480, 240),   # Middle right
                (320, 360),   # Bottom left
                (400, 360),   # Bottom middle
                (480, 360),   # Bottom right
            ]
        ]
        
        return [DotPattern(p) for p in patterns]
            
    def read(self, camera_index: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Generate a synthetic frame with static dots."""
        if camera_index is not None:
            return self._generate_frame(camera_index)
        else:
            frames = []
            timestamps = []
            for i in range(self.num_cameras):
                frame, timestamp = self._generate_frame(i)
                frames.append(frame)
                timestamps.append(timestamp)
            return frames, timestamps
    
    def _generate_frame(self, camera_index: int) -> Tuple[np.ndarray, float]:
        """Generate a single synthetic frame with bright white dots."""
        # Create base frame (dark background)
        if self._colour:
            frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        else:
            frame = np.zeros((self._height, self._width), dtype=np.uint8)
            
        # Draw dots
        pattern = self._patterns[camera_index]
        for x, y in pattern.positions:
            # Create a bright white dot
            dot_size = 8  # Large dots for better visibility
            dot = np.zeros((dot_size*2+1, dot_size*2+1), dtype=np.uint8)
            cv2.circle(dot, (dot_size, dot_size), dot_size-2, 255, -1)
            
            # Place dot in frame
            y1, y2 = max(0, y-dot_size), min(self._height, y+dot_size+1)
            x1, x2 = max(0, x-dot_size), min(self._width, x+dot_size+1)
            dy1, dy2 = max(0, dot_size-y), dot_size*2+1 - max(0, (y+dot_size+1)-self._height)
            dx1, dx2 = max(0, dot_size-x), dot_size*2+1 - max(0, (x+dot_size+1)-self._width)
            
            if self._colour:
                # Make dots pure white (255 for all channels)
                white_dot = np.full((dot[dy1:dy2, dx1:dx2].shape[0], 
                                   dot[dy1:dy2, dx1:dx2].shape[0], 3), 255, dtype=np.uint8)
                mask = dot[dy1:dy2, dx1:dx2] > 0
                frame[y1:y2, x1:x2][mask] = white_dot[mask]
            else:
                frame[y1:y2, x1:x2] = dot[dy1:dy2, dx1:dx2]
        
        # Apply exposure and gain with higher base brightness
        frame = frame * (self._gain[camera_index]/16) * (self._exposure[camera_index]/64)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        return frame, time.time()
    
    @property
    def exposure(self) -> List[int]:
        return self._exposure
    
    @exposure.setter
    def exposure(self, values: List[int]):
        if isinstance(values, list):
            self._exposure = values
        else:
            self._exposure = [values] * self.num_cameras
    
    @property
    def gain(self) -> List[int]:
        return self._gain
    
    @gain.setter
    def gain(self, values: List[int]):
        if isinstance(values, list):
            self._gain = values
        else:
            self._gain = [values] * self.num_cameras
            
    def end(self):
        """Clean up resources."""
        pass