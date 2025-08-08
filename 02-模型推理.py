from ultralytics import YOLO
import os

# 创建保存结果的目录
os.makedirs('results', exist_ok=True)

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 设置测试图片文件夹路径
test_folder = 'datasets/ball/test/images'

# 遍历测试文件夹中的所有图片
for filename in os.listdir(test_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):  # 支持常见的图片格式
        # 构造完整的图片路径
        image_path = os.path.join(test_folder, filename)
        
        # 运行预测
        results = model(image_path)
        
        # 构造保存的文件名
        save_filename = os.path.join('results', f'predicted_{filename}')
        
        # 保存预测结果
        results[0].save(filename=save_filename)
        print(f'预测结果已保存到 {save_filename}')