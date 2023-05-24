import numpy as np

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
from matplotlib.figure import Figure

import threading
import queue
import time
import yaml

import tkinter
from tkinter import ttk
import cv2
from CameraHandler import CameraHandler

class TkInterface:
  """
  Instanciate class, then call mainloop() method
  """
  def __init__(self, cam_handler: CameraHandler):
    self._cam_handler = cam_handler
    self._window = tkinter.Tk()

    self._window.protocol("WM_DELETE_WINDOW", self.pre_exiting)
    self._window.geometry("800x600")
    self._window.wm_title("Embedding in Tk")

    self._btn_exit = ttk.Button(self._window, text="Exit", command=self.pre_exiting)
    self._btn_exit.grid(row=0, column=0)

    self._tabs_ctrl = ttk.Notebook(self._window)

    # ================================================================================================
    # Controls tab
    self._control_tab = ttk.Frame(self._tabs_ctrl)

    # Defined here, but put in grid later
    self._contrast = ContrastPlot(self._control_tab, self._cam_handler)

    self._basic_processing = BasicProcessing(self._control_tab, self._cam_handler, self._contrast)
    self._basic_processing.grid(row=1, column=0, rowspan=1, columnspan=1)

    self._image_integration = ImageIntegration(self._control_tab, self._cam_handler)
    self._image_integration.grid(row=2, column=0, rowspan=1, columnspan=1)

    self._contrast.grid(row=3, column=0, rowspan=1, columnspan=1)

    # ================================================================================================
    # File save tab
    self._files_tab = ttk.Frame(self._tabs_ctrl)
    self._save = Save(self._files_tab, self._cam_handler)
    self._save.grid(row=0, column=0)

    # ================================================================================================
    # Parameters setup tab
    self._config_tab = ttk.Frame(self._tabs_ctrl)
    self._setup = Setup(self._config_tab, self._cam_handler)
    self._setup.grid(row=0, column=0)

    # Add tabs to layout
    self._tabs_ctrl.add(self._control_tab, text="Control")
    self._tabs_ctrl.add(self._files_tab, text="Saving")
    self._tabs_ctrl.add(self._config_tab, text="Config")
    self._tabs_ctrl.grid(row=1, column=0)

  def pre_exiting(self):
    self._contrast.stop_rendering()
    self._window.after(100, self._exiting)

  def _exiting(self):
    # Close contrast hist update thread (must have called stop_rendering method before and let tkinter run some time)
    print("Stop contrast plot thread...")
    self._contrast.stop()

    # Close opencv thread
    print("Stop cam handling thread and window...")
    self._cam_handler.stop()

    print("Stop tkinter...")
    # Then stop tkinter mainloop and threads
    self._window.quit()
    self._window.destroy()

  def mainloop(self):
    self._contrast.start() # start hist plot update thread
    tkinter.mainloop()

class ContrastPlot:
  """
  Instanciate, then call grid() method to add to window
  """
  def __init__(self, parent, cam_handler: CameraHandler):
    self._parent = parent
    self._cam_handler = cam_handler
    self._thread_stop_queue = queue.Queue(maxsize=32)
    self._data_queue = queue.Queue(maxsize=32)

    self._frame: ttk.LabelFrame = ttk.LabelFrame(self._parent, text="Contrast Histogram")

    # Local save of last data plotted
    self._X = np.arange(0, 256, 1)
    self._Y = np.zeros((256, 1))

    self._gate = np.ones((256, 1))

    # Plot update period
    self._s_per_frames = 1.0 / 10.0 # 10 FPS

    # Create plotting figures:
    with plt.style.context("dark_background"):
      self._fig = Figure(figsize=(8, 4), dpi=100)

      ax_hist = self._fig.add_subplot(111)
      ax_hist.title.set_text("Image intensity histogram")
      self._plot = ax_hist.plot(self._X, self._Y)
      self._plot += ax_hist.plot(self._X, self._gate)

      self._plot[0].axes.set_ylim([0,1])

    # Create canvas in which matplotlib figure is drawn
    self._canvas = FigureCanvasTkAgg(self._fig, master=self._frame)  # A tk.DrawingArea.
    self._canvas.draw()
    self._canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, padx=0, pady=0)

    # Create thread (not started yet)
    self._thread = threading.Thread(target=self._thread_update_function, args=(self._thread_stop_queue, self._data_queue))

  def grid(self, row=0, column=0, rowspan=0, columnspan=0):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan)

  def start(self):
    if not self._thread.is_alive():
      self._thread.start()

  def stop_rendering(self):
    """
    Queues a command to the plot update thread to stop adding deferred rendering to tkinter canvas.
    Necessary to call this, let some tkinter window updates go by and then call stop method, otherwise the app might freeze.
    """
    self._thread_stop_queue.put("PauseRender")

  def stop(self):
    """
    Stops update thread if is running. Blocking until thread is closed.
    """
    if self._thread.is_alive():
      self._thread_stop_queue.put("Exit")
      self._thread.join()

  def update_gate(self, gate: tuple):
    if not self._data_queue.full():
      self._data_queue.put(gate)

  def update_data(self, data: np.array):
    if not self._data_queue.full():
      self._data_queue.put(data)

  def _thread_update_function(self, control_queue: queue.Queue, data_queue: queue.Queue):
    """
    Thread function.
    """
    print("Thread plots update started.")
    pause_render = False
    running = True
    (gate_low, gate_high) = (0, 255)
    while running:
      # Get latest data to plot from queue
      while not data_queue.empty():
        (gate_low, gate_high) = data_queue.get()
        for i in range(len(self._gate)):
          if gate_low <= i and i < gate_high:
            self._gate[i] = 100000000.0
          else:
            self._gate[i] = 0.0
        self._plot[1].set_data(self._X, self._gate)

      if not control_queue.empty():
        cmd = control_queue.get()
        if cmd == "PauseRender":
          pause_render = True
        if cmd == "Exit":
          print("Stopping hist update thread...")
          running = False

      data = self._cam_handler.get_last_integrated_img()
      if data is not None:
        hist, _ = np.histogram(data.flatten(), 256, [0, 256])
        self._Y = hist
        self._Y[0] = 0
        self._plot[0].set_data(self._X, self._Y)

      # Update axes
      self._plot[0].axes.set_ylim([np.min(self._Y) - 1,np.max(self._Y) + 1])


      if not pause_render:
        self._canvas.draw()

      time.sleep(self._s_per_frames)

    print("Thread hist update stopped.")

class BasicProcessing:
  """
  Instanciate, then call grid() method to add to window
  """
  def __init__(self, parent, cam_handler: CameraHandler, contrast_plot: ContrastPlot):
    self._parent = parent
    self._cam_handler = cam_handler
    self._contrast_plot = contrast_plot

    self._frame: ttk.LabelFrame = ttk.LabelFrame(self._parent, text="Basic Processing")

    # ########################################################################################################
    # Dark/Noise image handling
    self._btn_dark_img = ttk.Button(self._frame, text="Take dark image", command=self._take_dark_img)
    self._btn_dark_img.grid(row=0, column=0)

    self._check_dark_value = tkinter.BooleanVar()
    self._check_dark = ttk.Checkbutton(
      self._frame,
      text="Subtract",
      variable=self._check_dark_value,
      onvalue=True,
      offvalue=False,
      command=self._check_dark_update
    )
    self._check_dark.grid(row=0, column=1)

    # ########################################################################################################
    # Overlay handling
    self._btn_static_img = ttk.Button(self._frame, text="Take static image", command=self._take_static_img)
    self._btn_static_img.grid(row=1, column=0)

    self._check_static_value = tkinter.BooleanVar()
    self._check_static = ttk.Checkbutton(
      self._frame,
      text="Overlay",
      variable=self._check_static_value,
      onvalue=True,
      offvalue=False,
      command=self._check_static_update
    )
    self._check_static.grid(row=1, column=1)

    ttk.Label(self._frame, text="Overlay opacity: ").grid(row=1, column=2)
    self._scale_static = ttk.Scale(self._frame, from_=0, to=100, command=self._scale_static_update)
    self._scale_static.set(50)
    self._scale_static.grid(row=1, column=3)
    # ########################################################################################################
    # Contrast thresholds handling
    self._gate_low_label = ttk.Label(self._frame, text="Low contrast gate: 0")
    self._gate_low_label.grid(row=2, column=0)
    self._scale_gate_low = ttk.Scale(self._frame, from_=0, to=254, command=self._gate_scale_update, length=200)
    self._scale_gate_low.grid(row=2, column=1, columnspan=3)

    self._gate_high_label = ttk.Label(self._frame, text="High contrast gate: 1")
    self._gate_high_label.grid(row=3, column=0)
    self._scale_gate_high = ttk.Scale(self._frame, from_=1, to=255, command=self._gate_scale_update, length=200)
    self._scale_gate_high.set(255)
    self._scale_gate_high.grid(row=3, column=1, columnspan=3)
    # ########################################################################################################

  def grid(self, row: int, column: int, rowspan: int, columnspan: int):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan)

  def _take_dark_img(self):
    if self._cam_handler.get_last_img() is None:
      print("Warning: could not save dark image because no snapped image.")
    else:
      self._cam_handler.update_param("dark", self._cam_handler.get_last_img())
      cv2.imwrite('images/dark.bmp', self._cam_handler.get_last_img())

  def _take_static_img(self):
    if self._cam_handler.get_last_img() is None:
      print("Warning: could not save static image because no snapped image.")
    else:
      self._cam_handler.update_param("static", self._cam_handler.get_last_img())
      cv2.imwrite('images/static.bmp', self._cam_handler.get_last_img())

  def _check_dark_update(self):
    self._cam_handler.update_param("subtract_dark", self._check_dark_value.get())

  def _check_static_update(self):
    self._cam_handler.update_param("overlay_static", self._check_static_value.get())

  def _scale_static_update(self, _):
    self._cam_handler.update_param("overlay_opacity", self._scale_static.get())

  def _gate_scale_update(self, _):
    gate_low = self._scale_gate_low.get()
    gate_high = self._scale_gate_high.get()

    # Prevent incorrect gate
    if gate_low >= gate_high:
      gate_high = gate_low + 1
      self._scale_gate_high.set(gate_high)

    self._gate_low_label.config(text = f"Low contrast gate: {int(gate_low)}")
    self._gate_high_label.config(text = f"High contrast gate: {int(gate_high)}")

    # Send new gate to contrast plot update thread queue
    self._contrast_plot.update_gate((gate_low, gate_high))
    self._cam_handler.update_param("gate", (gate_low, gate_high))

class ImageIntegration:
  """
  Instanciate, then call grid() method to add to window
  """
  def __init__(self, parent, cam_handler: CameraHandler):
    self._parent = parent
    self._cam_handler = cam_handler

    self._frame: ttk.LabelFrame = ttk.LabelFrame(self._parent, text="Image Integration")

    # ########################################################################################################
    # Image integration handling

    self._scale_label_val = tkinter.StringVar()
    self._scale_label_val.set("1")
    self._scale_label = ttk.Label(self._frame, textvariable=self._scale_label_val)
    self._scale_label.grid(row=0, column=0)

    self._scale_integr = ttk.Scale(self._frame, from_=1, to=50, command=self._integr_val_update)
    self._scale_integr.grid(row=0, column=1)

    self._check_integr_value = tkinter.BooleanVar()
    self._check_integr = ttk.Checkbutton(
      self._frame,
      text="Average",
      variable=self._check_integr_value,
      onvalue=True,
      offvalue=False,
      command=self._check_integr_update
    )
    self._check_integr.grid(row=0, column=2)

    # ########################################################################################################

  def grid(self, row: int, column: int, rowspan: int, columnspan: int):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan)

  def _integr_val_update(self, _):
    val = int(self._scale_integr.get())
    self._cam_handler.update_param("integration_val", val)
    self._scale_label_val.set(val)

  def _check_integr_update(self):
    self._cam_handler.update_param("integration", self._check_integr_value.get())

class Save:
  """
  Instanciate, then call grid() method to add to window
  """
  def __init__(self, parent, cam_handler: CameraHandler):
    self._parent = parent
    self._cam_handler = cam_handler

    self._frame: ttk.LabelFrame = ttk.LabelFrame(self._parent, text="Image save")

    # ########################################################################################################
    # Image saving

    self._prefix_label = ttk.Label(self._frame, text="Saved image prefix: ")
    self._prefix_label.grid(row=0, column=0)

    self._prefix_entry_value = tkinter.StringVar()
    self._prefix_entry = ttk.Entry(self._frame, textvariable=self._prefix_entry_value, width=50)
    self._prefix_entry.grid(row=0, column=1)

    self._save_btn = ttk.Button(self._frame, text="Save", command=self._save_btn_update)
    self._save_btn.grid(row=1, column=0)

    self._save_confirm_label = ttk.Label(self._frame, text="")
    self._save_confirm_label.grid(row=1, column=1)

  def grid(self, row: int, column: int, rowspan: int = 1, columnspan: int = 1):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan)

  def _save_btn_update(self):
    prefix = self._prefix_entry_value.get()
    if prefix == "":
      prefix = "default"
    filename = "images/" + prefix + ".bmp"
    try:
      cv2.imwrite(filename, self._cam_handler.get_last_final_img())
      self._save_confirm_label.config(text = f"SAVED AS ./{filename} !")
      self._parent.after(1000, self._save_confirm_clear)
    except Exception as err:
      self._save_confirm_label.config(text = f"ERROR ! {err}")
      self._parent.after(1000, self._save_confirm_clear)

  def _save_confirm_clear(self):
    self._save_confirm_label.config(text = "")

class Setup:
  """
  Instanciate, then call grid() method to add to window
  """
  def __init__(self, parent, cam_handler: CameraHandler):
    self._parent = parent
    self._cam_handler = cam_handler

    self._frame: ttk.LabelFrame = ttk.LabelFrame(self._parent, text="Camera Parameters Setup")

    # ########################################################################################################
    # Parameters list
    self._params_frame = ttk.LabelFrame(self._frame, text="Parameters from yaml config file")
    self._params_frame.grid(row=0, column=0, columnspan=2)

    # First time yaml config file load and send to MMCore
    self._params_widgets = []
    self._yaml_config: dict = {}
    self._reload_config()

    # For each entry in yaml file, create an editable entry widget on GUI
    row = 0
    for k_device, v_device in self._yaml_config.items():
      device_frame = ttk.LabelFrame(self._params_frame, text=k_device)
      device_frame.grid(row=row, column=0)
      row += 1
      subrow = 0
      for k, v in v_device.items():
        label = ttk.Label(device_frame, text=k_device + "." + k)
        param_value = tkinter.StringVar()
        param_value.set(v)
        param_entry = ttk.Entry(device_frame, textvariable=param_value)

        label.grid(row=subrow, column=0)
        param_entry.grid(row=subrow, column=1)
        subrow += 1

        self._params_widgets.append((label, param_value))

    self._reload_btn = ttk.Button(self._frame, text="Reload config", command=self._reload_config)
    self._reload_btn.grid(row=1, column=0)

    self._save_btn = ttk.Button(self._frame, text="Save and send", command=self._update_config)
    self._save_btn.grid(row=1, column=1)

  def grid(self, row: int, column: int, rowspan: int = 1, columnspan: int = 1):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan)

  def _update_config(self):
    for (label, param_value) in self._params_widgets:
      device_name = label["text"].split(".")[0]
      param_name = label["text"].split(".")[1]
      self._yaml_config[device_name][param_name] = param_value.get()

      self._cam_handler.update_camera_parameter(device_name, param_name, param_value.get())

    with open("parameters.yml", "w", encoding="utf-8") as f:
      yaml.dump(self._yaml_config, f)

  def _reload_config(self):
    self._yaml_config = {}
    with open("parameters.yml", "r", encoding="utf-8") as f:
      self._yaml_config = yaml.safe_load(f)

    for k_device, v_device in self._yaml_config.items():
        for k, v in v_device.items():
          self._cam_handler.update_camera_parameter(k_device, k, v)

    for (label, param_value) in self._params_widgets:
      param_value.set(self._yaml_config[label["text"].split(".")[0]][label["text"].split(".")[1]])
