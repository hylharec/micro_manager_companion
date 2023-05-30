import numpy as np

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)
from matplotlib.figure import Figure

import os, threading, queue, time, yaml

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
    self._window.geometry("820x740")
    self._window.wm_title("Embedding in Tk")

    self._btn_exit = ttk.Button(self._window, text="Exit", command=self.pre_exiting)
    self._btn_exit.grid(row=0, column=0, padx=5, pady=5)

    self._tabs_ctrl = ttk.Notebook(self._window)

    # ================================================================================================
    # Controls tab
    self._control_tab = ttk.Frame(self._tabs_ctrl)

    # Defined here, but put in grid later
    self._contrast = ContrastPlot(self._control_tab, self._cam_handler)

    self._autogui = AutoGui(self._control_tab, self._cam_handler, "gui.yml")
    self._autogui.grid(row=0, column=0)

    #self._basic_processing = BasicProcessing(self._control_tab, self._cam_handler, self._contrast)
    #self._basic_processing.grid(row=1, column=0, rowspan=1, columnspan=1)
    #
    #self._image_integration = ImageIntegration(self._control_tab, self._cam_handler)
    #self._image_integration.grid(row=2, column=0, rowspan=1, columnspan=1)

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

    # ================================================================================================
    # AutoGui tab
    #self._autogui_tab = ttk.Frame(self._tabs_ctrl)
    #self._autogui = AutoGui(self._autogui_tab, self._cam_handler, "gui.yml")
    #self._autogui.grid(row=0, column=0)

    # ================================================================================================

    # Add tabs to layout
    self._tabs_ctrl.add(self._control_tab, text="Control")
    self._tabs_ctrl.add(self._files_tab, text="Saving")
    self._tabs_ctrl.add(self._config_tab, text="Config")
    #self._tabs_ctrl.add(self._autogui_tab, text="AutoGui")
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

    # Create images subdirectory if does not exist already (to save/load images)
    os.makedirs("images", exist_ok=True)

    # Local save of last data plotted
    self._X = np.arange(0, self._cam_handler.BIT_DEPTH, 1)
    self._Y = np.zeros((self._cam_handler.BIT_DEPTH, 1))

    self._gate = np.ones((self._cam_handler.BIT_DEPTH, 1))

    # Plot update period
    self._s_per_frames = 1.0 / 30.0 # 10 FPS

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

  def grid(self, row=0, column=0, rowspan=0, columnspan=0, padx: int = 5, pady: int = 5):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=padx, pady=pady)

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
    (gate_low, gate_high) = (0, self._cam_handler.BIT_DEPTH-1)
    while running:
      # Get latest data to plot from queue
      if not data_queue.empty():
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
        hist, _ = np.histogram(data.flatten(), self._cam_handler.BIT_DEPTH, [0, self._cam_handler.BIT_DEPTH])
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
    self._btn_dark_img.grid(row=0, column=0, padx=5, pady=5)

    self._check_dark_value = tkinter.BooleanVar()
    self._check_dark = ttk.Checkbutton(
      self._frame,
      text="Subtract",
      variable=self._check_dark_value,
      onvalue=True,
      offvalue=False,
      command=self._check_dark_update
    )
    self._check_dark.grid(row=0, column=1, padx=5, pady=5)

    self._check_dark_mode_value = tkinter.BooleanVar()
    self._check_dark_mode = ttk.Checkbutton(
      self._frame,
      text="Absolute Diff",
      variable=self._check_dark_mode_value,
      onvalue=True,
      offvalue=False,
      command=self._check_dark_mode_update
    )
    self._check_dark_mode.grid(row=0, column=2, padx=5, pady=5)

    # ########################################################################################################
    # Overlay handling
    self._btn_static_img = ttk.Button(self._frame, text="Take static image", command=self._take_static_img)
    self._btn_static_img.grid(row=1, column=0, padx=5, pady=5)

    self._check_static_value = tkinter.BooleanVar()
    self._check_static = ttk.Checkbutton(
      self._frame,
      text="Overlay",
      variable=self._check_static_value,
      onvalue=True,
      offvalue=False,
      command=self._check_static_update
    )
    self._check_static.grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(self._frame, text="Overlay opacity: ").grid(row=1, column=2, padx=5, pady=5)
    self._scale_static = ttk.Scale(self._frame, from_=0, to=100, command=self._scale_static_update)
    self._scale_static.set(50)
    self._scale_static.grid(row=1, column=3, padx=5, pady=5)
    # ########################################################################################################
    # Contrast thresholds handling
    self._gate_low_label = ttk.Label(self._frame, text="Low contrast gate: 0")
    self._gate_low_label.grid(row=2, column=0, padx=5, pady=5)
    self._scale_gate_low = ttk.Scale(self._frame, from_=1, to=self._cam_handler.BIT_DEPTH-2, command=self._gate_scale_update, length=300)
    self._scale_gate_low.grid(row=2, column=1, columnspan=3, padx=5, pady=5)

    self._gate_high_label = ttk.Label(self._frame, text="High contrast gate: 1")
    self._gate_high_label.grid(row=3, column=0, padx=5, pady=5)
    self._scale_gate_high = ttk.Scale(self._frame, from_=2, to=self._cam_handler.BIT_DEPTH-1, command=self._gate_scale_update, length=300)
    self._scale_gate_high.grid(row=3, column=1, columnspan=3, padx=5, pady=5)

    # Set default values
    self._scale_gate_low.set(1)
    self._scale_gate_high.set(self._cam_handler.BIT_DEPTH-1)
    # ########################################################################################################

  def grid(self, row: int, column: int, rowspan: int, columnspan: int, padx: int = 5, pady: int = 5):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=padx, pady=pady)

  def _take_dark_img(self):
    if self._cam_handler.get_last_equalized_img() is None:
      print("Warning: could not save dark image because no snapped image.")
    else:
      self._cam_handler.update_param("dark", self._cam_handler.get_last_equalized_img())
      cv2.imwrite('images/dark.png', self._cam_handler.get_last_equalized_img())

  def _take_static_img(self):
    if self._cam_handler.get_last_equalized_img() is None:
      print("Warning: could not save static image because no snapped image.")
    else:
      self._cam_handler.update_param("static", self._cam_handler.get_last_equalized_img())
      cv2.imwrite('images/static.png', self._cam_handler.get_last_equalized_img())

  def _check_dark_update(self):
    self._cam_handler.update_param("subtract_dark", self._check_dark_value.get())

  def _check_dark_mode_update(self):
    self._cam_handler.update_param("subtraction_mode", self._check_dark_mode_value.get())

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
    self._cam_handler.update_param("gate_low", gate_low)
    self._cam_handler.update_param("gate_high", gate_high)

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
    self._scale_label.grid(row=0, column=0, padx=5, pady=5)

    self._scale_integr = ttk.Scale(self._frame, from_=1, to=50, command=self._integr_val_update)
    self._scale_integr.grid(row=0, column=1, padx=5, pady=5)

    self._check_integr_value = tkinter.BooleanVar()
    self._check_integr = ttk.Checkbutton(
      self._frame,
      text="Average",
      variable=self._check_integr_value,
      onvalue=True,
      offvalue=False,
      command=self._check_integr_update
    )
    self._check_integr.grid(row=0, column=2, padx=5, pady=5)

    # ########################################################################################################

  def grid(self, row: int, column: int, rowspan: int, columnspan: int, padx: int = 5, pady: int = 5):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=padx, pady=pady)

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
    self._prefix_label.grid(row=0, column=0, padx=5, pady=5)

    self._prefix_entry_value = tkinter.StringVar()
    self._prefix_entry = ttk.Entry(self._frame, textvariable=self._prefix_entry_value, width=50)
    self._prefix_entry.grid(row=0, column=1, padx=5, pady=5)

    self._save_btn = ttk.Button(self._frame, text="Save", command=self._save_btn_update)
    self._save_btn.grid(row=1, column=0, padx=5, pady=5)

    self._save_confirm_label = ttk.Label(self._frame, text="")
    self._save_confirm_label.grid(row=1, column=1, padx=5, pady=5)

  def grid(self, row: int, column: int, rowspan: int = 1, columnspan: int = 1, padx: int = 5, pady: int = 5):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=padx, pady=pady)

  def _save_btn_update(self):
    prefix = self._prefix_entry_value.get()
    if prefix == "":
      prefix = "default"
    filename = "images/" + prefix + ".png"
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
    self._params_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

    # First time yaml config file load and send to MMCore
    self._params_widgets = []
    self._yaml_config: dict = {}
    self._load_config_file()

    # For each entry in yaml file, create an editable entry widget on GUI
    row = 0
    for k_device, v_device in list(self._yaml_config.values())[0].items():
      device_frame = ttk.LabelFrame(self._params_frame, text=k_device)
      device_frame.grid(row=row, column=0, padx=5, pady=5)
      row += 1
      subrow: int = 0
      for k, v in v_device.items():
        setup_entry = SetupEntry(device_frame, k_device, k, v)
        setup_entry.grid(row=subrow, column=0)
        subrow += 1

        self._params_widgets.append(setup_entry)

    # Drop down menu that enables preset switching, on first preset by default
    presets = list(self._yaml_config.keys())
    self._preset_list_value = tkinter.StringVar()
    self._preset_list = ttk.OptionMenu(self._frame, self._preset_list_value, presets[0], *presets, command=self._preset_list_update)
    self._preset_list.grid(row=1, column=0, padx=5, pady=5)

    # Sends config to mm after drop down menu was created
    self._send_config_to_mm()

    self._reload_btn = ttk.Button(self._frame, text="Load", command=self._reload_config)
    self._reload_btn.grid(row=1, column=1, padx=5, pady=5)

    self._save_btn = ttk.Button(self._frame, text="Save", command=self._save_config)
    self._save_btn.grid(row=2, column=0, padx=5, pady=5)

    self._apply_btn = ttk.Button(self._frame, text="Apply", command=self._apply_config)
    self._apply_btn.grid(row=2, column=1, padx=5, pady=5)

  def grid(self, row: int, column: int, rowspan: int = 1, columnspan: int = 1, padx: int = 5, pady: int = 5):
    self._frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=padx, pady=pady)

  def _preset_list_update(self, _):
    """
    Automatically reloads config from file to GUI when selecting a new preset value in drop down menu.
    """
    self._reload_config()

  def _apply_config(self):
    """
    Sends current GUI config changes to micro manager.
    (Warning: Does not save to file)
    """
    # Reads config changes from GUI
    self._read_config_from_gui()
    # Sends new config to micro manager
    self._send_config_to_mm()

  def _save_config(self):
    """
    Reads config changes from GUI and saves them to file
    """
    # Read GUI
    self._read_config_from_gui()
    # Save to file
    self._save_config_file()

    # Set all entries color to green
    for entry in self._params_widgets:
      entry.green()

  def _reload_config(self):
    """
    Reloads yaml config file and propagates new values to GUI.
    """
    # Reloads from file
    self._load_config_file()

    current_preset = self._preset_list_value.get()

    # Update GUI depending on current preset
    for setup_entry in self._params_widgets:
      device_name = setup_entry.device_name
      param_name = setup_entry.label["text"]
      setup_entry.value.set(self._yaml_config[current_preset][device_name][param_name])

    # Set all entries color to green
    for entry in self._params_widgets:
      entry.green()

  def _load_config_file(self):
    """
    Reads yaml config file and stores as Python dictionary.
    """
    self._yaml_config = {}
    with open("parameters.yml", "r", encoding="utf-8") as f:
      self._yaml_config = yaml.safe_load(f)

  def _save_config_file(self):
    """
    Saves config dictionary to yaml file.
    """
    # Dumps dictionnary in yaml config file
    with open("parameters.yml", "w", encoding="utf-8") as f:
      yaml.dump(self._yaml_config, f)

  def _read_config_from_gui(self):
    """
    Reads GUI to update changes in config dictionary.
    """
    preset_name = self._preset_list_value.get()
    # Reads from Entry widgets to update params dictionary
    for setup_entry in self._params_widgets:
      device_name = setup_entry.device_name
      param_name = setup_entry.label["text"]

      self._yaml_config[preset_name][device_name][param_name] = setup_entry.value.get()

  def _send_config_to_mm(self):
    """
    Sends the config in dictionary to micro manager.
    (Warning: Does not read config file or GUI, needs to be done before calling this method)
    """
    preset_name = self._preset_list_value.get()
    for device_name, device_config in self._yaml_config[preset_name].items():
      for param_name, param_value in device_config.items():
        self._cam_handler.update_camera_parameter(device_name, param_name, param_value)

class SetupEntry:
  """
  Small class used to simplify handling of automatically generated parameters configuration menu from yaml file.
  """
  def __init__(self, parent, device_name: str, param_name: str, param_value):
    self._entry_saved_style = ttk.Style()
    self._entry_saved_style.configure("saved_entry.TEntry", foreground="green")
    self._entry_unsaved_style = ttk.Style()
    self._entry_unsaved_style.configure("unsaved_entry.TEntry", foreground="red")

    self.device_name = device_name
    self.label = ttk.Label(parent, text=param_name)
    self.value = tkinter.StringVar()
    self.value.set(param_value)
    self._param_entry = ttk.Entry(
      parent,
      textvariable=self.value,
      validate="focusout",
      validatecommand=self._entry_update,
      style="saved_entry.TEntry"
    )

  def grid(self, row: int, column: int):
    self.label.grid(row=row, column=0, padx=5, pady=5)
    self._param_entry.grid(row=row, column=1, padx=5, pady=5)

  def green(self):
    """
    Switches entry forground to green
    """
    self._param_entry.config(style = "saved_entry.TEntry")

  def _entry_update(self):
    self._param_entry.config(style = "unsaved_entry.TEntry")
    return False

class Filters:
  """
  Instanciate, then call grid() method to add to window
  """
  def __init__(self, parent, cam_handler: CameraHandler):
    self._parent = parent
    self._cam_handler = cam_handler

    self._frame: ttk.LabelFrame = ttk.LabelFrame(self._parent, text="Post-processing Filters")

    # ########################################################################################################
    # Applying filters on the final image

    # ########################################################################################################

class AutoGui:
  def __init__(self, parent, cam_handler: CameraHandler, yaml_file_r_path: str):
    self._parent = parent
    self._cam_handler = cam_handler

    self._frame = ttk.Frame(self._parent)

    with open(yaml_file_r_path) as f:
      self._yaml = yaml.safe_load(f)

    self._tabs = []
    tabs_names = []
    self._notebook = ttk.Notebook(self._frame)
    self._widgets = []
    for tab, content in self._yaml.items():
      if type(content) == dict:
        self._tabs.append(ttk.Frame(self._notebook))
        tabs_names.append(tab)
        row = 0
        for line, line_content in content.items():
          column = 0
          if type(line_content) == dict:
            for widget, params in line_content.items():
              if type(params) == dict:
                if params["type"] == "scale":
                  w = AutoGuiScale(
                    self._tabs[-1],
                    self._cam_handler,
                    from_=int(params["from"]),
                    to=int(params["to"]),
                    default=int(params["default"]),
                    length=int(params["length"]),
                    span=int(params["span"]),
                    name=widget,
                    param_name=params["param"]
                  )
                  w.grid(row=row, column=column)
                  self._widgets.append(w)
                  column += int(params["span"])
                elif params["type"] == "check":
                  w = AutoGuiCheck(
                    self._tabs[-1],
                    self._cam_handler,
                    default=False,
                    name=widget,
                    param_name=params["param"]
                  )
                  w.grid(row=row, column=column)
                  self._widgets.append(w)
                  column += 1
                elif params["type"] == "button":
                  w = AutoGuiButton(
                    self._tabs[-1],
                    self._cam_handler,
                    function=params["function"],
                    name=widget,
                    param_name=params["param"]
                  )
                  w.grid(row=row, column=column)
                  self._widgets.append(w)
                  column += 1
            row += 1
        self._tabs[-1].grid(row=0, column=0)
    for tab, name in zip(self._tabs, tabs_names):
      self._notebook.add(tab, text=name)
    self._notebook.grid(row=0, column=0)

  def grid(self, row, column):
    self._frame.grid(row=row, column=column)

class AutoGuiScale:
  def __init__(self, parent, cam_handler, from_: int, to: int, default: int, length: int, span: int, name: str, param_name: str):
    self._parent = parent
    self._cam_handler = cam_handler
    self._from_ = from_
    self._to = to
    self._default = default
    self._length = length
    self._span = span
    self.name = name
    self.param_name = param_name

    self._frame = ttk.Frame(parent)

    self._label = ttk.Label(self._frame, text=name + ": ")
    self._label.grid(row=0, column=0, padx=5, pady=5)

    self._widget = ttk.Scale(self._frame, from_=from_, to=to, length=length, command=self._update)
    self._widget.grid(row=0, column=1, padx=5, pady=5)

    self._label_value = ttk.Label(self._frame, text=f"({default})")
    self._label_value.grid(row=0, column=2, padx=5, pady=5)

    self._widget.set(default)

  def grid(self, row, column):
    self._frame.grid(row=row, column=column, columnspan=self._span, padx=5, pady=5)

  def _update(self, _):
    val = self._widget.get()
    self._cam_handler.update_param(self.param_name, val)
    self._label_value.config(text = f"({int(val)})")

class AutoGuiCheck:
  def __init__(self, parent, cam_handler, default: bool, name: str, param_name: str):
    self._parent = parent
    self._cam_handler = cam_handler
    self._default = default
    self.name = name
    self.param_name = param_name

    self._widget_value = tkinter.BooleanVar()
    self._widget_value.set(default)
    self._widget = ttk.Checkbutton(
      self._parent,
      text=name,
      variable=self._widget_value,
      onvalue=True,
      offvalue=False,
      command=self._update
    )

  def grid(self, row, column):
    self._widget.grid(row=row, column=column, padx=5, pady=5)

  def _update(self):
    self._cam_handler.update_param(self.param_name, self._widget_value.get())

class AutoGuiButton:
  def __init__(self, parent, cam_handler, function, name: str, param_name: str):
    self._parent = parent
    self._cam_handler = cam_handler
    self._function = function
    self.name = name
    self.param_name = param_name

    self._widget = ttk.Button(self._parent, text=name, command=self._update)

  def grid(self, row, column):
    self._widget.grid(row=row, column=column, padx=5, pady=5)

  def _update(self):
    getattr(self._cam_handler, self._function)()


class FilterEntry:
  def __init__(self, parent, cam_handler: CameraHandler, name: str, filter_function):
    self._parent = parent
    self._cam_holder = cam_handler

    self._frame = ttk.Frame(self._parent)

    self.filter_function = filter_function

    self._name = name
    self._check = ttk.Checkbutton(
      self._frame,
      text=name,
      variable=self._check_integr_value,
      onvalue=True,
      offvalue=False,
      command=self._check_update
    )
    self._check.grid(row=0, column=0)

  def grid(self, row: int, column: int, padx: int = 5, pady: int = 5):
    self._frame.grid(row=row, column=column, padx=padx, pady=pady)

  def _check_update(self):
    self._cam_holder.update_param("filter_" + self._name, (self._check.get(), self.filter_function))
