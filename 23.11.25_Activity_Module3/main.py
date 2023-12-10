def convert_shoe_size(size_string):
  """
  Converts a shoe size string into mm.

  Args:
    size_string: The shoe size string, e.g. "6.5 MED", "5.5 WIDE", "US 7.5", "LARGE", "Mens 6.0"

  Returns:
    The shoe size in mm, or None if the size cannot be parsed.
  """

  size, width, system = split_shoe_size(size_string)
  if system == "US":
    size_in_mm = convert_us_shoe_size(size, width)
  elif system == "UK":
    size_in_mm = convert_uk_shoe_size(size, width)
  elif system == "Womens":
    size_in_mm = convert_womens_shoe_size(size)
  elif system in ("MEDIUM", "SMALL", "LARGE"):
    size_in_mm = convert_size_label(size)
  else:
    raise ValueError(f"Unknown shoe size system: {system}")

  return size_in_mm

def split_shoe_size(size_string):
  """
  Splits a shoe size string into its components: size, width, and system.

  Args:
    size_string: The shoe size string, e.g. "6.5 MED", "5.5 WIDE", "US 7.5", "LARGE", "Mens 6.0"

  Returns:
    A tuple containing the size, width, and system: (float, str, str)
  """

  size_string_upper = size_string.upper()
  if "MED" in size_string_upper:
    size, width, system = size_string_upper.split(" ")
    system = "MEDIUM"
  elif "WIDE" in size_string_upper:
    size, width, system = size_string_upper.split(" ")
    system = "WIDE"
  else:
    # assume system is US, UK, Womens, or a size label
    size = size_string_upper.split(" ")[0]
    width = ""
    system = size_string_upper.split(" ")[-1]

  size = float(size)
  return size, width, system

def convert_us_shoe_size(size, width):
  """
  Converts a US shoe size to mm.

  Args:
    size: The US shoe size (float)
    width: The shoe width (str)

  Returns:
    The shoe size in mm (int)
  """

  size_in_inches = size + 1.5 if width == "W" else size + 1.0
  size_in_mm = int(round(size_in_inches * 25.4))
  return size_in_mm

def convert_uk_shoe_size(size, width):
  """
  Converts a UK shoe size to mm.

  Args:
    size: The UK shoe size (float)
    width: The shoe width (str)

  Returns:
    The shoe size in mm (int)
  """

  size_in_inches = size + 1.17 if width == "E" else size + 0.85
  size_in_mm = int(round(size_in_inches * 25.4))
  return size_in_mm

def convert_womens_shoe_size(size):
  """
  Converts a Womens shoe size to mm.

  Args:
    size: The Womens shoe size (float)

  Returns:
    The shoe size in mm (int)
  """

  size_in_inches = size + 1.43
  size_in_mm = int(round(size_in_inches * 25.4))
  return size_in_mm

def convert_size_label(size_label):
  """
  Converts a size label ("SMALL", "MEDIUM", "LARGE") to mm.

  Args:
    size_label: The size label (str)

  Returns:
    The shoe size in mm (int)
  """

  size_mapping = {
      "SMALL": 230,
      "MEDIUM": 250,
      "LARGE": 270,
  }
  return size_mapping[size_label]

# Example usage
shoe_size_strings = ["6.5 MED", "5.5 WIDE", "US 7.5", "LARGE", "Mens 6.0"]
for size_string in shoe_size_strings:
  try:
    size_in_mm = convert_shoe_size(size_string)
    print(f"Shoe size: {size_string}, mm: {size_in_mm}")
  except ValueError as e:
    print(f"Error: {e}")


