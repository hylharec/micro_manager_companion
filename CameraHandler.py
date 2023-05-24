import sys, queue, threading
import time
import cv2
import numpy as np
from pycromanager import Core

class CameraHandler:
    def __init__(self):
      # Connect to MM _core server
      try:
        self._core: Core = Core()
      except Exception:
        print("Could not connect to MicroManager core server, make sure it is running. Exit.\n\n")
        exit()

      self._update_thread_queue = queue.Queue(maxsize=32)
      self._snap_thread_queue = queue.Queue(maxsize=32)
      self._parameters_queue = queue.Queue(maxsize=32)
      self._update_thread = threading.Thread(target=self._update, args=(self._update_thread_queue, self._parameters_queue))
      self._snap_thread = threading.Thread(target=self._snap_update, args=(self._snap_thread_queue,))
      self._img_queue = []
      self._last_integrated_img = None
      self._last_final_img = None
      self._MAX_INTEGRATION_LEN = 50

      do_mirror_x = int(self._core.get_property('OpenCVgrabber', 'Flip X'))
      self._core.set_property('OpenCVgrabber', 'Flip X', 1 - do_mirror_x)
      self._core.set_property("OpenCVgrabber", "Exposure", "500")

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
        "gate": (0, 255),
      }

      cv2.namedWindow("cv_win", cv2.WINDOW_NORMAL)
      cv2.startWindowThread()

      while True:
        # Safely update parameters from queue (hypothesis: low param update rate)
        while not param_queue.empty():
          (key, val) = param_queue.get()
          params[key] = val

        if len(self._img_queue) > 0:
          # Last received image is at the end of the queue
          result = np.array(self._img_queue[-1])

          # Integrate over last few images if required
          if params["integration"] is True:
            nb_images = min(params["integration_val"], len(self._img_queue))
            for i in range(nb_images):
              result = cv2.addWeighted(result, i / nb_images, self._img_queue[-i-1], (nb_images - i) / nb_images, 0)

          self._last_integrated_img = np.array(result)

          # Apply noise image substraction if required
          if params["dark"] is not None and params["subtract_dark"] is True:
            if params["subtraction_mode"] == "absdiff":
              result = cv2.absdiff(result, params["dark"])
            else:
              result = cv2.subtract(result, params["dark"])

          # Apply equalization before adding the overlay

          hist,_ = np.histogram(result.flatten(),256,[0,256])
          (gate_low, gate_high) = params["gate"]
          lut = np.zeros((256, 1))
          for i in range(256):
            if i < gate_low:
              lut[i] = 0
            elif i < gate_high:
              lut[i] = (i - gate_low) / (gate_high - gate_low) * 255
            else:
              lut[i] = 255

          for x in range(len(result)):
            result[x] = lut[result[x]].reshape((result.shape[1]))

          #result = lut[result]
          self._last_equilized_img = result

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
        tagged_image = self._core.get_tagged_image()
        pixels = np.reshape(
            tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"], 4]
        )
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

        # Save raw image
        cv2.imwrite('images/snap.bmp', pixels)

        result = pixels

        # Queue last received image
        self._img_queue.append(result)

        # Pop oldest image if max queue length was reached
        if(len(self._img_queue) == self._MAX_INTEGRATION_LEN + 1):
          self._img_queue.pop(0)

        # Arbitrary sleep time, technically not necessary as "self._core_snap_image()" is blocking
        time.sleep(0.005)

        if not control_queue.empty() and control_queue.get() == "Exit":
          break

    def update_param(self, key, val):
      self._parameters_queue.put((key, val))
