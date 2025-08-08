from ultralytics import YOLO
import cv2

# 1. 加载训练好的YOLOv11模型
# 替换为你的模型路径（如.pt文件），若使用预训练模型可直接写'models/yolov11n.pt'（n/s/m/l/x版本）
model = YOLO(r"runs\detect\train\weights\best.pt")

# 2. 配置视频源和输出
input_video_path = "input_video_2.mp4"  # 输入视频路径
output_video_path = "output_video_2.mp4"  # 输出视频路径

# 3. 读取视频并获取基本信息
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度

# 4. 设置视频写入器（编码格式推荐使用mp4v）
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 5. 逐帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取完毕
    
    # 6. 使用YOLOv11进行检测
    # conf: 置信度阈值（如0.5，过滤低置信度结果）
    # iou: IOU阈值（非极大值抑制参数）
    results = model(frame, conf=0.5, iou=0.45)
    
    # 7. 可视化检测结果（在帧上绘制边界框和类别）
    annotated_frame = results[0].plot()  # 直接调用plot()方法生成带标注的帧
    
    # 8. 显示实时结果（可选，用于调试）
    cv2.imshow("YOLOv11 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
        break
    
    # 9. 写入输出视频
    out.write(annotated_frame)

# 10. 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"检测完成，结果保存至：{output_video_path}")
