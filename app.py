import gradio as gr
import torch
from run import init_model
from train_eval import predict  # 导入预测函数

# 初始化配置和模型
model, config = init_model()  # 确保初始化完成
# 加载训练好的模型权重（确保路径正确）
model.load_state_dict(torch.load("THUCNews/saved_dict/bert.ckpt", map_location='cpu'))
model.to(config.device)

en_cn_mapping = {
    "finance": "财经",
    "realty": "房地产",
    "stocks": "股票",
    "education": "教育",
    "science": "科技",
    "society": "社会",
    "politics": "政治",
    "sports": "体育",
    "game":"游戏",
    "entertainment": "娱乐",
}
# 创建界面
def classify_text(text):
    result = predict(config, model, text)

    # 直接返回概率字典（数值保持float类型）
    cn_probs = {
        en_cn_mapping.get(k, "未知"): float(v)  # 保持数值类型
        for k, v in result["probabilities"].items()
    }

    # 返回元组：文本结果 + 概率字典
    return (
        en_cn_mapping.get(result["pred_label"], "未知"),  # 文本结果
        cn_probs  # 概率字典
    )

interface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=3, placeholder="输入新闻文本..."),
    outputs=[
        gr.Text(label="预测类别"),  # 单独显示文本结果
        gr.Label(label="概率分布", num_top_classes=3)  # 仅显示数值概率
    ],
    title="中文新闻分类(BERT)",
    examples=[["中国队夺得世界杯冠军"], ["科技公司发布新一代人工智能芯片"]],
)

if __name__ == "__main__":
    interface.launch()