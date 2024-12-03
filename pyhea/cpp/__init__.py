try:
    from .accelerate import *
except ImportError as e:
    print(f"Warning: Failed to import accelerate module: {e}")
    print("This might happen if the C++ extension was not properly built.")
    raise