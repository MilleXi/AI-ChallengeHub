from ultralytics import YOLO
model=YOLO('yolov8n.pt')
results = model('https://ultralytics.com/images/bus.jpg')

# 打印检测结果详情
for result in results:
    boxes = result.boxes  # 边界框输出
    print(boxes.xyxy)     # 边界框坐标(xyxy格式)
    print(boxes.conf)     # 置信度分数
    print(boxes.cls)      # 类别ID
