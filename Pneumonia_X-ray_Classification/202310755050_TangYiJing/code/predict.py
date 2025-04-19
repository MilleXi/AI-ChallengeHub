from tensorflow.keras.preprocessing import image
import numpy as np

def predict_pneumonia(img_path, model):
    # 加载并预处理图像
    img = image.load_img(img_path, target_size=(150, 150))  # 调整尺寸需与训练时一致
    x = image.img_to_array(img)                             # 转为数组
    x = np.expand_dims(x, axis=0) / 255.                   # 添加批次维度并归一化

    # 预测
    pred_prob = model.predict(x)[0][0]  # 获取sigmoid输出的概率值
    pred_class = "肺炎" if pred_prob > 0.5 else "正常"
    
    return pred_class, float(pred_prob)  # 返回类别和概率

# 使用示例
img_path = "/content/data/chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg"
class_label, prob = predict_pneumonia(img_path, model)
print(f"预测结果: {class_label} (置信度: {prob:.2%})")
