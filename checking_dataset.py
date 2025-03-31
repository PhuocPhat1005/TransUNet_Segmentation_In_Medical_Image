import os

def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

def scan_files(directory):
    file_count = 0
    file_summary = {} # tổng hợp số files theo định dạng

    print(f'Scanning files in directory: {directory}\n')

    for root, dirs, files in os.walk(directory):
        if files:
            print(f"Directory: {root}")
        for file in files:
            file_count += 1
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file)[-1].lower()

            if file_ext not in file_summary:
                file_summary[file_ext] = 0

            file_summary[file_ext] += 1

            print(f" {file_path} - Size: {sizeof_fmt(file_size)} - Extension: {file_ext}")

    print("\nSummary:")
    print(f"Total files found: {file_count}")
    for ext, count in file_summary.items():
        print(f"  Extension '{ext}': {count} file(s)")

def main():
    download_path = "data/synapse/raw/"


    if not os.path.exists(download_path):
        print(f"Directory '{download_path}' does not exist. Vui lòng kiểm tra lại đường dẫn.")
        return

    scan_files(download_path)

if __name__ == "__main__":
    main()