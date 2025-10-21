
import sys

# not used
def get_multiline_input_until_eof():
    """
    Reads multiline input from the user until an EOF signal is received.
    Returns the concatenated multiline string.
    """
    sys.stdin.flush()
    print("Enter your multiline text (press Ctrl+D or Ctrl+Z then Enter to finish):")
    lines = sys.stdin.readlines()
    # Remove trailing newlines from each line if desired
    # lines = [line.rstrip('\n') for line in lines]
    sys.stdin.flush()
    return "".join(lines)

def get_multiline_input():
    """
    Reads multiline input from the user until an EOF signal is received.
    Returns the concatenated multiline string.
    """
    sys.stdin.flush()
    print("""Enter your multiline text: 
             (type EOF as last line then Enter to finish)
             (type exit to exit)
             """)
    lines = []
    while True:
        line = input()
        if line.strip() == "EOF" or line.strip() == "eof":
            break
        lines.append(line)
        if line.strip() == "exit":
            return "exit"
    # Remove trailing newlines from each line if desired
    # lines = [line.rstrip('\n') for line in lines]
    sys.stdin.flush()
    return "\n".join(lines)

def contains_mostly_numbers(s, threshold=0.2):
    """
    Checks if a string contains mostly numeric characters based on a given threshold.

    Args:
        s (str): The input string.
        threshold (float): The minimum proportion of numeric characters required
                           for the string to be considered "mostly numbers" (0.0 to 1.0).

    Returns:
        bool: True if the proportion of numeric characters meets or exceeds the threshold,
              False otherwise.
    """
    if not s:  # Handle empty string case
        return False

    numeric_count = 0
    for char in s:
        if char.isdigit():  # Checks if the character is a digit (0-9)
            numeric_count += 1

    proportion_numeric = numeric_count / len(s)
    return proportion_numeric >= threshold

