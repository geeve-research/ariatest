import argparse
import sys
import platform
import time
import numpy as np
import torch
import aria.sdk as aria
import rerun as rr
import cv2
from typing import List, Tuple, Sequence
from queue import Queue
import threading
from projectaria_tools.core.sensor_data import (
   ImageDataRecord,
   MotionData, 
   BarometerData
)
from projectaria_tools.core.mps import EyeGaze, get_eyegaze_point_at_depth
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import data_provider
from threading import Thread, Event
import queue

CALIBRATION_POINTS = [
   # Top row
   (200, 100), (600, 100), (960, 100), (1320, 100), (1720, 100),
   # Upper row  
   (200, 300), (600, 300), (960, 300), (1320, 300), (1720, 300),
   # Middle row
   (200, 540), (600, 540), (960, 540), (1320, 540), (1720, 540),
   # Lower row
   (200, 780), (600, 780), (960, 780), (1320, 780), (1720, 780),
   # Bottom row
   (200, 980), (600, 980), (960, 980), (1320, 980), (1720, 980),
]

GRID_NUMBERS = [
    (0, 0, "1"), (1, 0, "2"), (2, 0, "3"),
    (0, 1, "4"), (1, 1, "5"), (2, 1, "6")
]

class CalibrationPoint:
   def __init__(self, x: float, y: float, screen_x: int, screen_y: int):
       self.gaze_x = x
       self.gaze_y = y
       self.screen_x = screen_x
       self.screen_y = screen_y

class Calibrator:
   def __init__(self):
       self.calibration_points: List[CalibrationPoint] = []
       self.transform_matrix = None
       self.current_points = 0
       
   def add_point(self, gaze_x: float, gaze_y: float, screen_x: int, screen_y: int):
       self.calibration_points.append(CalibrationPoint(gaze_x, gaze_y, screen_x, screen_y))
       self.current_points += 1
       if self.current_points >= 4:
           self.compute_transform()
       
   def compute_transform(self):
       if len(self.calibration_points) < 4:
           return None

       gaze_coords = np.array([[p.gaze_x, p.gaze_y] for p in self.calibration_points])
       screen_coords = np.array([[p.screen_x, p.screen_y] for p in self.calibration_points])
       
       self.transform_matrix, _ = cv2.findHomography(gaze_coords, screen_coords)
       return self.transform_matrix

   def transform_gaze(self, gaze_x: float, gaze_y: float) -> Tuple[float, float]:
       if self.transform_matrix is None:
           screen_x = int(((gaze_x + 0.5) * 1920))
           screen_y = int(((gaze_y + 0.5) * 1080))
           return screen_x, screen_y

       point = np.array([[[gaze_x, gaze_y]]], dtype=np.float32)
       transformed = cv2.perspectiveTransform(point, self.transform_matrix)
       return transformed[0][0][0], transformed[0][0][1]

class CalibrationUI:
   def __init__(self, screen_width=1920, screen_height=1080):
       self.window_name = "Eye Tracking Calibration"
       self.screen_width = screen_width
       self.screen_height = screen_height
       self.calibration_points = CALIBRATION_POINTS
       self.current_point = 0
       self.calibrator = Calibrator()
       self.samples = []
       self.is_calibrating = False
       self.start_clicked = False
       self.grid_numbers_read = set()
       self.grid_cell_width = screen_width // 3
       self.grid_cell_height = screen_height // 2
       self.last_grid_number = None
       self.grid_dwell_time = {}
       self.required_dwell_time = 1.0  # 1 second dwell time required
       
   def draw_instruction_screen(self):
       screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
       font = cv2.FONT_HERSHEY_SIMPLEX
       text = "Press 'C' to start calibration"
       text_size = cv2.getTextSize(text, font, 1.5, 2)[0]
       text_x = (self.screen_width - text_size[0]) // 2
       text_y = self.screen_height // 2
       cv2.putText(screen, text, (text_x, text_y), font, 1.5, (255, 255, 255), 2)
       cv2.imshow(self.window_name, screen)

   def draw_grid_test_instruction(self):
       screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
       font = cv2.FONT_HERSHEY_SIMPLEX
       text = "Press 'G' to start grid test"
       text_size = cv2.getTextSize(text, font, 1.5, 2)[0]
       text_x = (self.screen_width - text_size[0]) // 2
       text_y = self.screen_height // 2
       cv2.putText(screen, text, (text_x, text_y), font, 1.5, (255, 255, 255), 2)
       cv2.imshow(self.window_name, screen)

   def draw_grid_screen(self, current_gaze=None):
       screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
       font = cv2.FONT_HERSHEY_SIMPLEX
       
       # Draw grid cells and numbers
       for grid_x, grid_y, number in GRID_NUMBERS:
           x1 = grid_x * self.grid_cell_width
           y1 = grid_y * self.grid_cell_height
           x2 = x1 + self.grid_cell_width
           y2 = y1 + self.grid_cell_height
           
           # Draw cell border
           cv2.rectangle(screen, (x1, y1), (x2, y2), (255, 255, 255), 2)
           
           # Calculate text position
           text_size = cv2.getTextSize(number, font, 4, 3)[0]
           text_x = x1 + (self.grid_cell_width - text_size[0]) // 2
           text_y = y1 + (self.grid_cell_height + text_size[1]) // 2
           
           # Highlight number if already read
           color = (0, 255, 0) if number in self.grid_numbers_read else (255, 255, 255)
           cv2.putText(screen, number, (text_x, text_y), font, 4, color, 3)

       # Draw current gaze point if available
       if current_gaze is not None:
           cv2.circle(screen, (int(current_gaze[0]), int(current_gaze[1])), 20, (0, 255, 0), 2)
           
       # Draw progress
       progress_text = f"Numbers read: {len(self.grid_numbers_read)}/6"
       cv2.putText(screen, progress_text, (20, 30), font, 1, (255, 255, 255), 2)
       
       cv2.imshow(self.window_name, screen)

   def draw_progress_screen(self):
       screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
       font = cv2.FONT_HERSHEY_SIMPLEX
       text = f"Calibrating: {self.current_point + 1}/{len(self.calibration_points)}"
       cv2.putText(screen, text, (50, 50), font, 1, (255, 255, 255), 2)
       return screen

   def draw_completion_screen(self):
       screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
       font = cv2.FONT_HERSHEY_SIMPLEX
       text = "Calibration Complete! Press 'G' to test calibration"
       text_size = cv2.getTextSize(text, font, 1.5, 2)[0]
       text_x = (self.screen_width - text_size[0]) // 2
       text_y = self.screen_height // 2
       cv2.putText(screen, text, (text_x, text_y), font, 1.5, (255, 255, 255), 2)
       cv2.imshow(self.window_name, screen)

   def get_grid_number(self, x, y):
       grid_x = int(x // self.grid_cell_width)
       grid_y = int(y // self.grid_cell_height)
       
       for gx, gy, number in GRID_NUMBERS:
           if gx == grid_x and gy == grid_y:
               return number
       return None

   def run_grid_test(self, observer):
       current_time = time.time()
       while len(self.grid_numbers_read) < 6:
           if len(observer.value_mapping) > 0:
               gaze_x = float(observer.value_mapping["yaw"])
               gaze_y = float(observer.value_mapping["pitch"])
               screen_x, screen_y = self.calibrator.transform_gaze(gaze_x, gaze_y)
               
               current_number = self.get_grid_number(screen_x, screen_y)
               
               if current_number:
                   if current_number not in self.grid_dwell_time:
                       self.grid_dwell_time[current_number] = time.time()
                   elif time.time() - self.grid_dwell_time[current_number] >= self.required_dwell_time:
                       self.grid_numbers_read.add(current_number)
               
               self.draw_grid_screen((screen_x, screen_y))
           
           key = cv2.waitKey(1) & 0xFF
           if key == 27:  # ESC
               return False
               
           if len(self.grid_numbers_read) == 6:
               screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
               font = cv2.FONT_HERSHEY_SIMPLEX
               text = "Grid test complete! Press 'S' to start program"
               text_size = cv2.getTextSize(text, font, 1.5, 2)[0]
               text_x = (self.screen_width - text_size[0]) // 2
               text_y = self.screen_height // 2
               cv2.putText(screen, text, (text_x, text_y), font, 1.5, (255, 255, 255), 2)
               cv2.imshow(self.window_name, screen)
               
               while True:
                   key = cv2.waitKey(1) & 0xFF
                   if key == ord('s'):
                       return True
                   elif key == 27:  # ESC
                       return False
       
       return True
       
   def run(self, observer, duration_per_point=3.0):
       cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
       cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
       
       self.draw_instruction_screen()
       
       while True:
           key = cv2.waitKey(1) & 0xFF
           if key == ord('c') and not self.is_calibrating:
               self.is_calibrating = True
               success = self.run_calibration(observer, duration_per_point)
               if success:
                   self.draw_grid_test_instruction()
               else:
                   break
           elif key == ord('g') and not self.is_calibrating and self.calibrator.transform_matrix is not None:
               if self.run_grid_test(observer):
                   self.start_clicked = True
                   break
               else:
                   break
           elif key == 27:  # ESC
               break
       
       cv2.destroyWindow(self.window_name)
       return self.start_clicked

   def run_calibration(self, observer, duration_per_point):
       font = cv2.FONT_HERSHEY_SIMPLEX
       last_update_time = 0
       update_interval = 0.05  # 50ms update interval
       
       for point_idx, point in enumerate(self.calibration_points):
           self.current_point = point_idx
           samples = []
           start_time = time.time()
           
           while time.time() - start_time < duration_per_point:
               current_time = time.time()
               screen = self.draw_progress_screen()
               
               # Draw target circle
               cv2.circle(screen, point, 40, (0, 0, 255), 2)  # Red circle
               cv2.circle(screen, point, 5, (0, 0, 255), -1)  # Red center dot
               
               # Draw live gaze point if available and enough time has passed
               if len(observer.value_mapping) > 0 and current_time - last_update_time >= update_interval:
                   gaze_x = float(observer.value_mapping["yaw"])
                   gaze_y = float(observer.value_mapping["pitch"])
                   samples.append((gaze_x, gaze_y))
                   
                   # Use recent samples for stable visualization
                   recent_samples = samples[-10:]
                   avg_x = np.mean([x for x, _ in recent_samples])
                   avg_y = np.mean([y for _, y in recent_samples])
                   
                   # Transform gaze using current calibration
                   gaze_screen_x, gaze_screen_y = self.calibrator.transform_gaze(avg_x, avg_y)
                   cv2.circle(screen, (int(gaze_screen_x), int(gaze_screen_y)), 20, (0, 255, 0), 2)
                   
                   last_update_time = current_time
               
               cv2.imshow(self.window_name, screen)
               
               key = cv2.waitKey(1)
               if key == 27:  # ESC
                   return False
           
           if samples:
               # Use last second of samples for calibration
               recent_samples = samples[-int(1.0/update_interval):]
               avg_x = np.mean([x for x, _ in recent_samples])
               avg_y = np.mean([y for _, y in recent_samples])
               self.calibrator.add_point(avg_x, avg_y, point[0], point[1])
       
       self.is_calibrating = False
       return True

class BaseStreamingClientObserver:
   def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
       pass

   def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
       pass

   def on_magneto_received(self, sample: MotionData) -> None:
       pass

   def on_baro_received(self, sample: BarometerData) -> None:
       pass

   def on_streaming_client_failure(self, reason, message: str) -> None:
       pass

class EyeTrackingObserver(BaseStreamingClientObserver):
   def __init__(self, inference_model, device_calibration, device="cpu"):
       self.inference_model = inference_model
       self.device_calibration = device_calibration
       self.device = device
       self.value_mapping = {}
       self.depth_m = 1
       self.T_device_CPF = device_calibration.get_transform_device_cpf()
       self.rgb_camera_calibration = device_calibration.get_camera_calib("camera-rgb")

   def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
       if record.camera_id == aria.CameraId.EyeTrack:
           image = np.ascontiguousarray(np.rot90(image, 2))
           img = torch.tensor(image, device=self.device)
           
           preds, lower, upper = self.inference_model.predict(img)
           preds = preds.detach().cpu().numpy()
           lower = lower.detach().cpu().numpy()
           upper = upper.detach().cpu().numpy()

           self.value_mapping = {
               "yaw": preds[0][0],
               "pitch": preds[0][1],
               "yaw_lower": lower[0][0],
               "pitch_lower": lower[0][1],
               "yaw_upper": upper[0][0],
               "pitch_upper": upper[0][1],
           }

class RealTimeVisualizer(BaseStreamingClientObserver):
   def __init__(self, inference_model, device_calibration, calibrator, device="cpu"):
       self.inference_model = inference_model
       self.device_calibration = device_calibration
       self.device = device
       self.calibrator = calibrator
       self.value_mapping = {}
       self.depth_m = 1
       self.T_device_CPF = device_calibration.get_transform_device_cpf()
       self.rgb_camera_calibration = device_calibration.get_camera_calib("camera-rgb")

       rr.log("device", rr.ViewCoordinates.RIGHT_HAND_X_DOWN, timeless=True)
       rr.log(
           "device/glasses_outline",
           rr.LineStrips3D(AriaGlassesOutline(device_calibration)),
           timeless=True,
       )
       
       self.color_mapping = {
           "yaw": [102, 255, 102],
           "pitch": [102, 178, 255],
           "yaw_lower": [102, 255, 102],
           "pitch_lower": [102, 102, 255],
           "yaw_upper": [102, 255, 178],
           "pitch_upper": [178, 102, 255],
       }
       for name, color in self.color_mapping.items():
           rr.log(
               f"eye_gaze_inference/{name}",
               rr.SeriesLine(color=color, name=name),
               timeless=True,
           )

   def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
       current_time = record.capture_timestamp_ns / 1e9
       rr.set_time_seconds("time", current_time)
       device_time_ns = record.capture_timestamp_ns
       rr.set_time_nanos("device_time", device_time_ns)
       rr.set_time_sequence("timestamp", device_time_ns)

       if record.camera_id == aria.CameraId.EyeTrack:
           image = np.ascontiguousarray(np.rot90(image, 2))
           img = torch.tensor(image, device=self.device)
           
           preds, lower, upper = self.inference_model.predict(img)
           preds = preds.detach().cpu().numpy()
           lower = lower.detach().cpu().numpy()
           upper = upper.detach().cpu().numpy()

           self.value_mapping = {
               "yaw": preds[0][0],
               "pitch": preds[0][1],
               "yaw_lower": lower[0][0],
               "pitch_lower": lower[0][1],
               "yaw_upper": upper[0][0],
               "pitch_upper": upper[0][1],
           }

           rr.log("camera-et", rr.Image(image))

           for name, value in self.value_mapping.items():
               rr.log(f"eye_gaze_inference/{name}", rr.Scalar(value))

           gaze_vector_in_cpf = get_eyegaze_point_at_depth(
               self.value_mapping["yaw"], 
               self.value_mapping["pitch"], 
               self.depth_m
           )
           
           rr.log(
               "device/eye-gaze",
               rr.Arrows3D(
                   origins=[self.T_device_CPF @ [0, 0, 0]],
                   vectors=[self.T_device_CPF @ gaze_vector_in_cpf],
                   colors=[[255, 0, 255]]
               ),
           )

       elif record.camera_id == aria.CameraId.Rgb:
           image = np.rot90(image, k=-1)
           rr.log("camera-rgb", rr.Image(image))

           if len(self.value_mapping) > 0:
               gaze_x = float(self.value_mapping["yaw"])
               gaze_y = float(self.value_mapping["pitch"])

               if self.calibrator and self.calibrator.transform_matrix is not None:
                   gaze_x, gaze_y = self.calibrator.transform_gaze(gaze_x, gaze_y)

               eye_gaze = EyeGaze
               eye_gaze.yaw = gaze_x
               eye_gaze.pitch = gaze_y
               
               gaze_projection = get_gaze_vector_reprojection(
                   eye_gaze,
                   "camera-rgb",
                   self.device_calibration,
                   self.rgb_camera_calibration,
                   self.depth_m
               )
               
               if gaze_projection is not None:
                   h, w = image.shape[:2]
                   x, y = gaze_projection
                   rotated_x = y
                   rotated_y = w - x
                   rotated_gaze_projection = np.array([rotated_x, rotated_y])
                   
                   rr.log(
                       "camera-rgb/eye-gaze_projection",
                       rr.Points2D(
                           rotated_gaze_projection,
                           radii=20,
                           colors=[[0, 255, 0]]
                       ),
                   )

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument(
       "--interface",
       dest="streaming_interface",
       type=str,
       required=True,
       choices=["usb", "wifi"],
   )
   parser.add_argument(
       "--model_checkpoint_path",
       type=str,
       required=True,
   )
   parser.add_argument(
       "--model_config_path",
       type=str,
       required=True,
   )
   parser.add_argument("--device", type=str, default="cpu")
   parser.add_argument("--device-ip", help="IP address for WiFi connection")
   parser.add_argument(
       "--vrs_file",
       type=str,
       default="44859ce1-717c-418f-be4c-ca5aad11e4e3.vrs",
   )
   return parser.parse_args()

def main():
   args = parse_args()
   
   from inference import infer
   inference_model = infer.EyeGazeInference(
       args.model_checkpoint_path, 
       args.model_config_path, 
       args.device
   )

   provider = data_provider.create_vrs_data_provider(args.vrs_file)
   device_calibration = provider.get_device_calibration()

   device_client = aria.DeviceClient()
   client_config = aria.DeviceClientConfig()
   
   if args.device_ip:
       client_config.ip_v4_address = args.device_ip
   device_client.set_client_config(client_config)
   
   device = device_client.connect()
   
   streaming_manager = device.streaming_manager
   streaming_client = streaming_manager.streaming_client
   
   streaming_config = aria.StreamingConfig()
   if args.streaming_interface == "usb":
       streaming_config.streaming_interface = aria.StreamingInterface.Usb
   streaming_config.security_options.use_ephemeral_certs = True
   
   streaming_manager.streaming_config = streaming_config
   streaming_manager.start_streaming()

   # First run calibration
   print("Starting calibration setup...")
   calibration_observer = EyeTrackingObserver(
       inference_model=inference_model,
       device_calibration=device_calibration,
       device=args.device
   )
   streaming_client.set_streaming_client_observer(calibration_observer)
   streaming_client.subscribe()

   calibration_ui = CalibrationUI()
   start_program = calibration_ui.run(calibration_observer)

   streaming_client.unsubscribe()

   if not start_program:
       print("Program cancelled")
       streaming_manager.stop_streaming()
       device_client.disconnect(device)
       return

   print("Starting visualization...")
   
   rr.init("Aria Eye Tracking", spawn=True)
   rr.set_time_seconds("time", 0)

   visualizer = RealTimeVisualizer(
       inference_model=inference_model,
       device_calibration=device_calibration,
       calibrator=calibration_ui.calibrator,
       device=args.device
   )
   
   streaming_client.set_streaming_client_observer(visualizer)
   streaming_client.subscribe()
   
   try:
       while True:
           time.sleep(0.001)
           
   except KeyboardInterrupt:
       print("\nStopping...")
   finally:
       streaming_client.unsubscribe()
       streaming_manager.stop_streaming()
       device_client.disconnect(device)

if __name__ == "__main__":
   main()