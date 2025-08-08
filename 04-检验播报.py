from ultralytics import YOLO
# import winsound
import pyttsx3

def set_voice():
    # 初始化
    engine = pyttsx3.init()
    # 设置声音：音量、类型、语速
    engine.setProperty('volume', 1.0)  # 音量
    engine.setProperty('rate', 150)  # 语速
    # 如果在 python 代码中遇到路径是 \，需要改成 \\ 或者 /
    engine.setProperty('voice','HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0')

    # 设置播放的内容
    info = '发现棒球'
    engine.say(info)
    # 执行
    engine.runAndWait()


if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')
    results = model.predict(
        source='datasets/ball/test/images/cricket_ball_3.jpg',
    )
    """
        提取出每一个框的信息：
            类别信息 置信度信息
        思路：
            1、定义一个列表，来存储每一条数据 one_list
            2、定义一个列表，存储所有的数据 result_list
    """
    result_list = []
    # 遍历处理结果
    for result in results:
        for item in result.boxes:
            # 类别信息
            cls = result.names[int(item.cls.item())]
            """
                如果发现球(cricketBall),进行播报
            """
            if cls == 'cricketBall':
                # 频率  持续时间
                # winsound.Beep(500, 2000)
                set_voice()
            # 置信度信息
            conf = round(item.conf.item(), 2)
            result_list.append([cls, conf])
    print(result_list)
