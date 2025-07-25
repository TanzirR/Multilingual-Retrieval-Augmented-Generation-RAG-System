import pickle
import pprint

file_path = 'chunks.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Successfully loaded '{file_path}'.")
    print(f"Type of loaded data: {type(data)}")
    print(f"Number of items (chunks) in the file: {len(data)}")

    # --- Inspecting the specific chunk (List Index 20) ---
    target_list_index = 20
    
    if len(data) > target_list_index:
        found_chunk_content = data[target_list_index]
        print(f"\n--- Full Content of Chunk at List Index {target_list_index} (likely CHUNK {target_list_index + 1}) ---")
        pprint.pprint(found_chunk_content)
        print("-" * 50)
        
        # --- Search within this specific chunk for confirmation ---
        if "পনেরো" in found_chunk_content or "১৫" in found_chunk_content:
            print(f"CONFIRMATION: 'পনেরো' or '১৫' FOUND in Chunk at List Index {target_list_index}!")
        else:
            print(f"CONFIRMATION: 'পনেরো' or '১৫' NOT FOUND in Chunk at List Index {target_list_index}'s text.")
    else:
        print(f"\nList index {target_list_index} is out of bounds for chunks.pkl (has {len(data)} items).")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the path.")
except pickle.UnpicklingError:
    print(f"Error: Could not unpickle the file '{file_path}'. It might be corrupted or not a valid pickle file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")