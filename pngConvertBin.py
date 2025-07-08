from PIL import Image
import numpy as np

# 打开 PNG 图像文件
image_path = 'Car.png'
image = Image.open(image_path)

# 将图像转换为 NumPy 数组
image_array = np.array(image)

# 将 NumPy 数组保存为二进制文件
output_path = 'car.bin'
image_array.tofile(output_path)

print(f"图像已成功转换为二进制文件并保存到 {output_path}")