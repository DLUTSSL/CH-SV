import os
import json
import torch
from tqdm import tqdm  # 进度条库
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 设置目录路径
video_dir = r"/root/autodl-tmp/视频/冒犯"  # 直接处理视频文件
output_json_path = "/root/autodl-tmp/test.json"  # 结果保存路径
failed_json_path = "/root/autodl-tmp/offensive_fail_all.json"  # 失败视频记录路径

# 加载模型和处理器
model_path = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# 提示词
PROMPTS = {
    "1": "视频主体指的是视频中最主要的角色、对象或元素。假如你是一个专业的视频分析员。基于给定的短视频，请定位该视频的主体，并以以下的格式输出该短视频的主体类型：{人类主体/其他主体}",
    "1-1": "该短视频以人类为主体，请从人类主体的外观、行为、情绪、语言四个角度对其进行详细描述与综合分析。请按照以下格式输出：{主体描述：“外观： ；行为： ；情绪： ；语言：”主体综合分析：}",
    "1-2": "该短视频主体并非是人类，请给出该短视频主体的详细描述与综合分析。请按照以下格式输出：{主体描述：“ ”主体综合分析：“ ”}",
    "2": "基于给定的短视频，判断该视频是否存在一个可以作为主线的事件。若存在明确的主线事件，请将其分类为‘事件型’；若没有明确的主线事件，且内容较为零散或不具备主线情节，请将其分类为‘非事件型’。输出格式{事件型/非事件型}",
    "2-1": "该短视频是事件型，请从事件的时间、地点、人物、起因、经过、结果这六个角度进行思考，给出对该事件的详细描述和综合分析。请按照以下格式输出：{内容描述：“时间：；地点：；人物：；事件起因：；事件经过：；事件结果：；”内容综合分析：}",
    "2-2": "该短视频是非事件型，请给出视频内容的详细描述和综合分析。请按照以下格式输出：{内容描述：内容综合分析：}",
}

# 读取已处理和失败的视频列表
def load_processed_videos():
    """加载已处理和失败的视频 ID，避免重复处理"""
    processed = set()
    failed = set()

    # 读取已处理视频
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                processed = {entry["video_id"] for entry in data}
            except json.JSONDecodeError:
                pass

    # 读取失败视频
    if os.path.exists(failed_json_path):
        with open(failed_json_path, "r", encoding="utf-8") as f:
            try:
                failed = set(json.load(f))
            except json.JSONDecodeError:
                pass

    return processed, failed

# 获取所有视频文件
def get_video_files(directory):
    """获取所有 MP4 格式的视频文件"""
    video_files = {}
    for filename in sorted(os.listdir(directory)):  # 确保顺序
        if filename.endswith(".mp4"):  # 只处理 MP4 视频
            video_id = os.path.splitext(filename)[0]  # 去掉后缀
            video_files[video_id] = os.path.join(directory, filename)
    return video_files

# 向 Qwen 提交请求
def query_qwen(video_path, prompt):
    """向 Qwen 提交请求，获取生成的文本"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "max_pixels": 180 * 210,
                    "fps": 1.0,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 预处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to("cuda")

    # 生成结果
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    return output_text

# 处理视频
def process_videos(video_files):
    """对每个视频进行完整的分析"""
    results = []
    processed_videos, failed_videos = load_processed_videos()
    failed_videos = set(failed_videos)  # 确保它是可变的集合
    total_videos = len(video_files)

    with tqdm(total=total_videos, desc="📊 视频分析进度", unit="video") as pbar:
        for video_id, video_path in video_files.items():
            if video_id in processed_videos:
                print(f"⏭️ 跳过 {video_id} (已完成)")
                pbar.update(1)
                continue  # 跳过已完成的视频
            
            if video_id in failed_videos:
                print(f"⚠️ 跳过 {video_id} (之前失败)")
                pbar.update(1)
                continue  # 跳过失败的视频

            print(f"\n🚀 Processing video: {video_id}")
            
            try:
                subject_type = query_qwen(video_path, PROMPTS["1"]).strip()
                if "人类主体" in subject_type:
                    subject_analysis = query_qwen(video_path, PROMPTS["1-1"])
                else:
                    subject_analysis = query_qwen(video_path, PROMPTS["1-2"])

                event_type = query_qwen(video_path, PROMPTS["2"]).strip()
                if "事件型" in event_type:
                    content_analysis = query_qwen(video_path, PROMPTS["2-1"])
                else:
                    content_analysis = query_qwen(video_path, PROMPTS["2-2"])

                video_result = {
                    "video_id": video_id,
                    "主体分析": subject_analysis,
                    "内容分析": content_analysis
                }
                results.append(video_result)

                # 追加新结果，而不是覆盖已有的
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"❌ 处理 {video_id} 失败: {e}")
                failed_videos.add(video_id)  # 更新失败的视频集合
                
                # 记录所有失败的视频
                with open(failed_json_path, "w", encoding="utf-8") as f:
                    json.dump(list(failed_videos), f, ensure_ascii=False, indent=4)

            pbar.update(1)

if __name__ == "__main__":
    video_files = get_video_files(video_dir)
    process_videos(video_files)
    print(f"\n✅ Analysis complete. Results saved to {output_json_path}")
