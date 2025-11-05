# zip_and_unzip.py
import zipfile
import os
import sys
import argparse

PRESETS = {
    "data": {
        "folder": "./datasets",     # zip/unzip the entire datasets tree
        "zip": "./datasets.zip",
    },
    "graph": {
        "folder": "./graphrag",
        "zip": "./graphrag.zip",
    },
    "weights": {
        "folder": "./weights",
        "zip": "./weights.zip",
    },
}

def zip_directory(folder_path, zip_path):
    if not os.path.isdir(folder_path):
        print(f"Error: folder not found: {folder_path}")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)) or ".", exist_ok=True)

    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=relative_path)
                count += 1
    print(f"✅ Zipped {count} files from '{folder_path}' into '{zip_path}'")

def _is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

def unzip_file(zip_path, extract_dir):
    if not os.path.isfile(zip_path):
        print(f"Error: zip file not found: {zip_path}")
        sys.exit(1)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zipf:
        for member in zipf.infolist():
            member_path = os.path.join(extract_dir, member.filename)
            if not _is_within_directory(extract_dir, member_path):
                print(f"Error: blocked unsafe path inside zip: {member.filename}")
                sys.exit(1)
        zipf.extractall(extract_dir)
    print(f"✅ Extracted '{zip_path}' to '{extract_dir}/'")

def main():
    parser = argparse.ArgumentParser(
        description="Zip/unzip using presets (data, graph, weights) or custom paths."
    )
    parser.add_argument("action", choices=["zip", "unzip"], help="zip or unzip")
    parser.add_argument("target", help="preset name (data, graph, weights) or 'custom'")
    # Custom:
    #   zip custom <folder_path> <zip_path>
    #   unzip custom <zip_path> <extract_dir>
    parser.add_argument("path1", nargs="?", help="custom: folder_path (zip) or zip_path (unzip)")
    parser.add_argument("path2", nargs="?", help="custom: zip_path (zip) or extract_dir (unzip)")
    args = parser.parse_args()

    if args.target == "custom":
        if args.action == "zip":
            if not args.path1 or not args.path2:
                print("Usage: python zip_and_unzip.py zip custom <folder_path> <zip_path>")
                sys.exit(2)
            zip_directory(args.path1, args.path2)
        else:
            if not args.path1 or not args.path2:
                print("Usage: python zip_and_unzip.py unzip custom <zip_path> <extract_dir>")
                sys.exit(2)
            unzip_file(args.path1, args.path2)
        return

    if args.target not in PRESETS:
        known = ", ".join(PRESETS.keys())
        print(f"Error: unknown target '{args.target}'. Known targets: {known}, or use 'custom'.")
        sys.exit(2)

    folder = PRESETS[args.target]["folder"]
    zip_path = PRESETS[args.target]["zip"]

    if args.action == "zip":
        zip_directory(folder, zip_path)
    else:
        unzip_file(zip_path, folder)

if __name__ == "__main__":
    main()

# Usage:
# python zip_and_unzip.py zip data
# python zip_and_unzip.py unzip data
# python zip_and_unzip.py zip graph
# python zip_and_unzip.py unzip graph
# python zip_and_unzip.py zip weights
# python zip_and_unzip.py unzip weights
# python zip_and_unzip.py zip custom ./some_folder ./some_folder.zip
# python zip_and_unzip.py unzip custom ./some_folder.zip ./restore_here
