import subprocess

def main():
    options = {
        "1": "test_graphrag_retriever.py",
        "2": "test_print_radgraph_jsons.py",
        "3": "test_reports.py"
    }

    print("Select which test code to run:")
    for k, v in options.items():
        print(f"{k}. {v}")

    choice = input("Enter 1/2/3/4: ").strip()
    if choice not in options:
        print("❌ Invalid choice.")
        return

    script = options[choice]
    script_path = f"test_codes/{script}"
    print(f"▶ Running {script_path}...\n")

    # Call the chosen script
    subprocess.run(["python", script_path])

if __name__ == "__main__":
    main()

# python run_test.py