import os

def delete_annotation_and_png_files(root_dir, delete_patterns):
    """
    Deletes files under the given directory based on name patterns.

    Args:
        root_dir (str): Path to the root directory to clean
        delete_patterns (dict): Dictionary with keys 'contains' and/or 'endswith',
                                each mapping to a list of substrings or suffixes.
                                Example:
                                {
                                    "contains": ["graph"],
                                    "endswith": [".png", "_graph.txt"]
                                }
    """
    deleted_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            should_delete = False

            # Check for 'contains' matches
            if "contains" in delete_patterns:
                if any(substr in filename for substr in delete_patterns["contains"]):
                    should_delete = True

            # Check for 'endswith' matches (only if not already matched)
            if not should_delete and "endswith" in delete_patterns:
                if any(filename.endswith(suffix) for suffix in delete_patterns["endswith"]):
                    should_delete = True

            if should_delete:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"🗑️ Deleted: {file_path}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {file_path}: {e}")

    print(f"\n✅ Deletion complete. Total files removed: {deleted_count}")
