import re

def transform_backlit(description):
    # Check for patterns and apply transformations
    if re.search(r'\b4[-\s]?zone\b', description, re.IGNORECASE):
        return "4-Zone RGB"
    elif re.search(r'\bbacklit\b', description, re.IGNORECASE):
        if re.search(r'\brgb\b', description, re.IGNORECASE):
            return "RGB"
        elif re.search(r'\bwhite\b', description, re.IGNORECASE):
            return "White"
        else:
            return "None"
    else:
        return "None"

# Test cases
print(transform_backlit("4-zone rgb backlit"))         # Output: 4-Zone RGB
print(transform_backlit("4 zone baclit"))              # Output: 4-Zone RGB
print(transform_backlit("backlit, rgb"))               # Output: RGB
print(transform_backlit("white backlit"))               # Output: White
print(transform_backlit("None backlit keyboard"))      # Output: None
print(transform_backlit("Random text"))                # Output: None
