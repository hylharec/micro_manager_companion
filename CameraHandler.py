import sys, queue, threading
import time
import cv2
import numpy as np
from pycromanager import Core

class CameraHandler:
    def __init__(self, verbose: bool = False):
      # Connect to MM _core server
      try:
        self._core: Core = Core()
      except Exception:
        print("Could not connect to MicroManager core server, make sure it is running. Exit.\n\n")
        exit()

      # Enables/Disables some print functions
      self._verbose = verbose

      self._update_thread_queue = queue.Queue(maxsize=32)
      self._snap_thread_queue = queue.Queue(maxsize=32)
      self._parameters_queue = queue.Queue(maxsize=32)
      self._update_thread = threading.Thread(target=self._update, args=(self._update_thread_queue, self._parameters_queue))
      self._snap_thread = threading.Thread(target=self._snap_update, args=(self._snap_thread_queue,))
      self._img_queue = []
      self._last_integrated_img = None
      self._last_equalized_img = None
      self._last_final_img = None
      self._MAX_INTEGRATION_LEN = 50

      # Set image grayscale format
      self._pixel_shape = 4 # 4 -> RGBA, 1 -> grayscale
      self._INPUT_BIT_DEPTH = 256#16384
      self.BIT_DEPTH = None
      self._image_dtype = None
      if np.log2(self._INPUT_BIT_DEPTH) > 32:
        self._image_dtype = np.uint64
        self.BIT_DEPTH = np.power(2, 64)
      elif np.log2(self._INPUT_BIT_DEPTH) > 16:
        self._image_dtype = np.uint32
        self.BIT_DEPTH = np.power(2, 32)
      elif np.log2(self._INPUT_BIT_DEPTH) >= 8: # TODO > instead of >=
        self._image_dtype = np.uint16
        self.BIT_DEPTH = np.power(2, 16)
      else:
        self._image_dtype = np.uint8
        self.BIT_DEPTH = np.power(2, 8)
      # Following attribute is used in picture snap thread function
      self._INPUT_TO_OUTPUT_BIT_DEPTH_MULT = int(np.power(2.0, (np.log2(self.BIT_DEPTH) - np.log2(self._INPUT_BIT_DEPTH))))

    def __del__(self):
      self.stop()

    def update_camera_parameter(self, device, key, val):
      self._core.set_property(device, key, val)

    def get_last_img(self):
      if len(self._img_queue) != 0:
        return self._img_queue[-1]
      else:
        return None

    def get_last_integrated_img(self):
      return self._last_integrated_img

    def get_last_equalized_img(self):
      return self._last_equalized_img

    def get_last_final_img(self):
      return np.array(self._last_final_img)

    def start(self):
      if not self._update_thread.is_alive():
        self._update_thread.start()
      if not self._snap_thread.is_alive():
        self._snap_thread.start()

    def stop(self):
      """
      Stops update thread if is running. Blocking until thread is closed.
      """
      if self._update_thread.is_alive():
        self._update_thread_queue.put("Exit")
        self._update_thread.join()

      if self._snap_thread.is_alive():
        self._snap_thread_queue.put("Exit")
        self._snap_thread.join()

    def _update(self, control_queue: queue.Queue, param_queue: queue.Queue):
      params = {
        "dark": None,
        "static": None,
        "subtract_dark": False,
        "overlay_static": False,
        "overlay_opacity": 50,
        "subtraction_mode": "subtract",
        "integration_val": 1,
        "integration": False,
        "gate_low": 1,
        "gate_high": self.BIT_DEPTH-1
      }

      cv2.namedWindow("cv_win", cv2.WINDOW_NORMAL)
      cv2.startWindowThread()

      # Try to load previously saved dark/static pictures (Error/Warning catching handled by opencv internally)
      params["dark"] = cv2.imread("images/dark.png", cv2.IMREAD_UNCHANGED)
      params["static"] = cv2.imread("images/static.png", cv2.IMREAD_UNCHANGED)

      # Some params are saved from one image processing loop to another to prevent recomputation
      # Default gate values (as wide as possible)
      gate_low, gate_high = params["gate_low"], params["gate_high"]
      # LUT Function to equalize image
      def lut_func(i: np.ndarray) -> np.ndarray:
        nonlocal self
        nonlocal gate_low
        nonlocal gate_high
        coef = (self.BIT_DEPTH - 1) / (gate_high - gate_low)
        return np.maximum(0, np.minimum(self.BIT_DEPTH-1, (i - gate_low) * coef))
      #lut_func_vec = np.vectorize(lut_func, otypes=[np.ndarray])

      while True:
        # Safely update parameters from queue (hypothesis: low param update rate)
        while not param_queue.empty():
          (key, val) = param_queue.get()
          params[key] = val

        if len(self._img_queue) > 0:
          # Last received image is at the end of the queue
          result = np.array(self._img_queue[-1])
          # If the snap thread works correctly, the image shape should be (width, height)

          # Integrate over last few images if required
          if params["integration"] is True:
            nb_images = min(int(params["integration_val"]), len(self._img_queue))
            for i in range(nb_images):
              result = cv2.addWeighted(result, i / nb_images, self._img_queue[-i-1], (nb_images - i) / nb_images, 0)

          # Apply noise image substraction if required
          if params["dark"] is not None and params["subtract_dark"] is True:
            if params["subtraction_mode"] is True:
              result = cv2.absdiff(result, params["dark"])
            else:
              result = cv2.subtract(result, params["dark"])

          self._last_integrated_img = np.array(result)

          # Apply equalization before adding the overlay
          (gate_low, gate_high) = params["gate_low"], params["gate_high"]

          result = lut_func(result).astype(self._image_dtype)

          #for x in range(len(result)):
          #  result[x] = lut[result[x]].reshape((result.shape[1]))

          #result = lut[result]
          self._last_equalized_img = np.array(result)

          if params["static"] is not None and params["overlay_static"] is True:
            # Send grayscale to red channel for better visualisation
            result = cv2.merge((result * 0, result * 0, result))
            static = cv2.cvtColor(params["static"], cv2.COLOR_GRAY2BGR)
            # Overlay
            overlay_opacity = params["overlay_opacity"] / 100.0
            result = cv2.addWeighted(result, 1.0 - overlay_opacity, static, overlay_opacity, 0)

          self._last_final_img = result
          cv2.imshow("cv_win", result)
          cv2.waitKey(20)

        # Handle thread stop command
        if not control_queue.empty() and control_queue.get() == "Exit":
          break

      cv2.destroyWindow("cv_win")

    def _snap_update(self, control_queue: queue.Queue):
      while True:
        self._core.snap_image()


        if self._verbose:
          print("Snap!")

        # Try might fail on the first line if MM fails to answer correctly
        try:
          tagged_image = self._core.get_tagged_image()

          pixels = np.reshape(
              tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"], self._pixel_shape]
          ).astype(self._image_dtype)

          if self._pixel_shape != 1:
            # Convert to grayscale if image is not already
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

          # Reshape to (_, _) in case shape was (_, _, 1)
          pixels = pixels.reshape((pixels.shape[0], pixels.shape[1]))

          # Apply values scalar multiplication in case bit depth is not a multiple of 2
          pixels = pixels * self._INPUT_TO_OUTPUT_BIT_DEPTH_MULT

          # Save raw image
          #cv2.imwrite('images/snap.bmp', pixels)

          result = pixels

          # Queue last received image
          self._img_queue.append(result)

          # Pop oldest image if max queue length was reached
          if(len(self._img_queue) == self._MAX_INTEGRATION_LEN + 1):
            self._img_queue.pop(0)

        except Exception:
          print("Warning: Error while getting image from MM. Ignoring...")

        # Arbitrary sleep time, technically not necessary as "self._core_snap_image()" is blocking
        time.sleep(0.005)

        if not control_queue.empty() and control_queue.get() == "Exit":
          break

    def update_param(self, key, val):
      self._parameters_queue.put((key, val))

    def take_dark_img(self):
      if self.get_last_equalized_img() is None:
        print("Warning: could not save dark image because no snapped image.")
      else:
        self.update_param("dark", self.get_last_equalized_img())
        cv2.imwrite('images/dark.png', self.get_last_equalized_img())

    def take_static_img(self):
      if self.get_last_equalized_img() is None:
        print("Warning: could not save static image because no snapped image.")
      else:
        self.update_param("static", self.get_last_equalized_img())
        cv2.imwrite('images/static.png', self.get_last_equalized_img())
