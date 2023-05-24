# local imports
from CameraHandler import CameraHandler
from TkInterface import TkInterface


if __name__ == "__main__":
  # Handles connection with MicroManager Core server and showing opencv window
  camHandler = CameraHandler(verbose=False)
  camHandler.start()

  # Handles tkinter GUI
  interface = TkInterface(camHandler)
  interface.mainloop()
