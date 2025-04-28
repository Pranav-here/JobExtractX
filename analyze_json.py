import json
import sys

try:
    # Open the file but only read a small portion
    with open("prepared_data_train.json", "r") as f:
        # Read first 10 characters to check format
        start = f.read(10)
        f.seek(0)
        
        if start.strip().startswith("["):
            print("File starts with an array")
            # Read until we get one complete item
            depth = 0
            in_string = False
            escape = False
            chars = []
            
            # Read enough to get first few items
            for i, c in enumerate(f.read(100000)):  # Read 100KB
                chars.append(c)
                
                if escape:
                    escape = False
                    continue
                    
                if c == "\\" and not escape:
                    escape = True
                    continue
                    
                if c == "\"" and not escape:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if c in "[{":
                        depth += 1
                    elif c in "]}":
                        depth -= 1
                        
                        # If we found at least one complete top-level item
                        if depth == 1 and c == "]" and chars.count("[") >= 3:
                            # We got at least one complete item
                            break
            
            # Parse what we have so far
            partial_json = "".join(chars)
            # Make sure it ends with a complete item
            if depth > 1:
                # Add closing brackets to complete the structure
                partial_json += "]" * (depth - 1)
                
            # Try to parse it
            try:
                data = json.loads(partial_json)
                print(f"Successfully parsed {len(data)} items from the beginning of the file")
                
                # Print sample of the first item
                if len(data) > 0:
                    print("\nFirst item structure:")
                    first_item = data[0]
                    print(f"Type: {type(first_item)}")
                    
                    if isinstance(first_item, list):
                        print(f"Item is a list with {len(first_item)} elements")
                        for i, element in enumerate(first_item):
                            print(f"Element {i} type: {type(element)}")
                            if isinstance(element, str):
                                print(f"Element {i} preview: {element[:100]}...")
                            else:
                                print(f"Element {i}: {element}")
                    else:
                        print(f"Item: {first_item}")
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print("First 200 characters:")
                print(partial_json[:200])
        else:
            print("File does not start with an array")
            print("First 100 characters:")
            print(start + f.read(90))
            
except Exception as e:
    print(f"Error: {e}")

