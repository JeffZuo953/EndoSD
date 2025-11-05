import torch
import sys

def print_keys_with_depth(data, current_depth, max_depth):
    """
    Recursively prints keys of nested dictionaries or elements up to a specified depth.

    Args:
        data (dict or any): The data structure to traverse.
        current_depth (int): The current depth of the traversal.
        max_depth (int): The maximum depth to print keys.
    """
    if current_depth > max_depth:
        return

    if isinstance(data, dict):
        for key, value in data.items():
            prefix = "  " * current_depth
            print(f"{prefix}- {key}")
            # Recursively call for nested dictionaries
            if isinstance(value, dict):
                print_keys_with_depth(value, current_depth + 1, max_depth)
            # If the value is not a dictionary but is an iterable (like a list or tuple
            # of tensors), we might want to indicate its presence without
            # printing all its contents, to stay within depth limits and avoid
            # excessive output. For depth=2, we just print the key of the list/tuple.
            # If we wanted to show keys within elements of a list/tuple,
            # we would need more complex logic and potentially increase max_depth.
            # For this script, we stop at depth 2.
            elif isinstance(value, (list, tuple)):
                 # Optionally print a placeholder for list/tuple content at depth 2
                 if current_depth + 1 <= max_depth:
                      list_item_prefix = "  " * (current_depth + 1)
                      # print(f"{list_item_prefix}(contains list/tuple items)") # Optional: uncomment to show list/tuple presence
                      pass # Do nothing further for list/tuple items at depth 2


    # We don't traverse into non-dict items like tensors at depth 2,
    # as the request is for keys of the structure.

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_pth_file>")
        sys.exit(1)

    pth_file_path = sys.argv[1]
    depth_limit = 2

    try:
        # Load the .pth file
        # Use map_location='cpu' to load to CPU to avoid potential CUDA issues
        checkpoint = torch.load(pth_file_path, map_location='cpu')

        print(f"Keys in '{pth_file_path}' up to depth {depth_limit}:")
        print_keys_with_depth(checkpoint, 0, depth_limit)

    except FileNotFoundError:
        print(f"Error: File not found at '{pth_file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)