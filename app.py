import gradio as gr
import numpy as np
import pandas as pd
import torch
from run import init_model
from train_eval import predict
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from matplotlib.font_manager import FontProperties
# 初始化模型
model, config = init_model()
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
    "game": "游戏",
    "entertainment": "娱乐",
}


def format_probabilities(probs):
    """将概率字典转换为可视化字符串"""
    return "\n".join([
        f"• {en_cn_mapping.get(k, '未知')}: {v*100:.2f}%"
        for k, v in sorted(probs.items(),
                         key=lambda x: x[1],
                         reverse=True)
    ])

# 文本分类处理函数
def text_classify(text):
    if not text.strip():
        return gr.Dataframe(visible=False),None  # 隐藏空结果

    result = predict(config, model, text)
    # 构建概率图表
    chart_fig = generate_probability_chart({
        en_cn_mapping.get(k, '未知'): v
        for k, v in result["probabilities"].items()
    }, title="📝 文本输入结果")
    # 构建扁平化数据结构
    df = pd.DataFrame([{
        "输入方式": "📝 文本输入",
        "内容/文件": text[:100] + "..." if len(text) > 100 else text,
        "预测结果": en_cn_mapping.get(result["pred_label"], "未知"),
        "概率分析": format_probabilities(result["probabilities"]),
        # "原始数据": result  # 调试用，实际部署时可移除
    }])
    return df,[chart_fig]


# 文件分类处理函数
def file_classify(files):
    if not files:
        return gr.Dataframe(visible=False),[]

    results = []
    chart_images = []
    for file in files:
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read().strip()
            result = predict(config, model, content)

            filename = file.name.split("/")[-1]  # 提取文件名

            results.append({
                "输入方式": "📁 文件输入",
                "内容/文件": filename,
                "预测结果": en_cn_mapping.get(result["pred_label"], "未知"),
                "概率分析": format_probabilities(result["probabilities"]),
                # "原始数据": result  # 调试用
            })
            # 将文件名传递给图表生成器
            chart_img = generate_probability_chart({
                en_cn_mapping.get(k, '未知'): v
                for k, v in result["probabilities"].items()
            }, title=filename)  # 核心改动点
            chart_images.append(chart_img)

    return pd.DataFrame(results), chart_images

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'FangSong']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    plt.rcParams['axes.facecolor'] = 'white'     # 设置背景色为白色

def generate_probability_chart(probabilities,title=""):
    setup_chinese_font()
    """生成概率分布柱状图"""
    # 过滤0概率类别
    probs = {k: v for k, v in probabilities.items() if v > 0}

    # 排序并截取top10
    sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10])

    # 创建图表
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.barh(list(sorted_probs.keys()), list(sorted_probs.values()))

    # 优化标签显示
    ax.set_xlabel('概率', fontsize=12)
    # 增强标题显示
    if title:
        ax.set_title(f'文件: {title}\n类别概率分布', fontsize=14, pad=20)
    else:
        ax.set_title('类别概率分布', fontsize=14, pad=20)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))  # 格式化x轴

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + 0.2, f'{width * 100:.1f}%', va='center')

    plt.tight_layout()

    # 转换为 numpy 数组
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer._renderer)
    plt.close()

    return img_array


# 创建带导航栏的界面
with gr.Blocks(title="中文新闻分类(BERT)") as demo:
    gr.Markdown("# 中文新闻分类系统（BERT模型）")

    with gr.Tabs():
        # 文本输入标签页
        with gr.Tab("📝 文本输入"):
            text_input = gr.Textbox(lines=5, placeholder="输入新闻文本...", label="新闻内容")
            text_button = gr.Button("开始分析")
            text_examples = gr.Examples(
                examples=[
                    ["中国队夺得世界杯冠军"],
                    ["央行宣布降准0.5个百分点"],
                    ["人工智能新突破：新型神经网络架构"]
                ],
                inputs=[text_input]
            )

        # 文件输入标签页
        with gr.Tab("📁 文件输入"):
            file_input = gr.File(file_count="multiple", file_types=[".txt"], label="上传新闻文件")
            file_button = gr.Button("开始分析")
            file_examples = gr.Examples(
                examples=[[["file1.txt", "file2.txt"]]],
                inputs=[file_input]
            )

    with gr.Column():
        output_table = gr.Dataframe(
            headers=["输入类型", "内容/文件名", "预测类别", "概率分布"],
            datatype=["str", "str", "str", "str"],
            label="分析结果",
            interactive=False,
            elem_id="result-table"
        )
        output_gallery = gr.Gallery(
            label="概率分布可视化",
            columns=2,
            height="auto",
            object_fit="contain",  # 保持比例完整显示
            visible=True)  # 修改为Gallery组件

    # 绑定事件处理
    text_button.click(
        fn=text_classify,
        inputs=text_input,
        outputs=[output_table, output_gallery]
    )

    file_button.click(
        fn=file_classify,
        inputs=file_input,
        outputs=[output_table, output_gallery]
    )

if __name__ == "__main__":
    demo.launch()