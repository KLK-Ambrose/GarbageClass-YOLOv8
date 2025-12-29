"""
垃圾分类训练脚本 - 使用 YOLOv8 分类模型
"""
from ultralytics import YOLO
from pathlib import Path

# ============== 配置区 ==============
DATA_ROOT = r"D:\code_files\GarbageClass\archive\garbage_classification"
MODEL_SIZE = "n"  # 可选: n(最快), s, m, l, x(最准)
EPOCHS = 100
IMG_SIZE = 224  # 分类任务常用 224
BATCH_SIZE = 32  # 根据显存调整


# ====================================

def main():
    # 加载预训练的分类模型
    model = YOLO(f"yolov8{MODEL_SIZE}-cls.pt")

    # 开始训练
    results = model.train(
        data=DATA_ROOT,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=20,  # 早停：20轮无提升则停止
        device=0,  # GPU 0，用 'cpu' 则使用CPU
        workers=4,  # 数据加载线程数
        project="runs/classify",
        name="garbage_cls",

        # 数据增强
        hsv_h=0.015,  # 色调变化
        hsv_s=0.7,  # 饱和度变化
        hsv_v=0.4,  # 亮度变化
        degrees=10,  # 旋转角度
        scale=0.5,  # 缩放比例
        fliplr=0.5,  # 水平翻转概率
    )

    print("\n训练完成！")
    print(f"最佳模型保存在: runs/classify/garbage_cls/weights/best.pt")


def predict_test():
    """测试训练好的模型"""
    model = YOLO("runs/classify/garbage_cls/weights/best.pt")

    # 预测单张图片
    results = model.predict(
        source=r"D:\code_files\GarbageClass\archive\garbage_classification\battery\battery1.jpg",
        save=True
    )

    # 打印预测结果
    for r in results:
        print(f"Top5 预测结果:")
        probs = r.probs
        top5_indices = probs.top5
        top5_conf = probs.top5conf
        names = r.names
        for idx, conf in zip(top5_indices, top5_conf):
            print(f"  {names[idx]}: {conf:.2%}")


if __name__ == "__main__":
    main()
    # predict_test()  # 训练完成后取消注释来测试