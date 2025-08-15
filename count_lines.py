import os

total_lines = 0

for folder, _, files in os.walk(r"E:\work\model4"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(folder, file)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                count = len(lines)
                total_lines += count
                print(f"{filepath}: {count} 行")

print(f"\n总行数: {total_lines} 行")
