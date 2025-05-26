import json
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import ast

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 强制将torch设置为使用第0号GPU
torch.cuda.set_device(0)

csv_path = '/mnt/ymj/vivo/地区异常/输出结果/异常检测结果.csv'  # 请将此处替换为你的实际文件路径
data = pd.read_csv(csv_path, encoding='utf-8')
print(f'总条数为{len(data)}')
print(f'本次解决的unlabel数量为{len(data)}')

def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)  # 添加 padding=True

    with torch.no_grad():  # 禁用梯度计算
        generated_ids = model.generate(model_inputs.input_ids, 
                                       attention_mask=model_inputs.attention_mask,  # 显式传递 attention_mask
                                       max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def clean_and_merge_response(response):
    # 保留换行符，只替换多余的回车符
    response = response.replace('\r', '')
    return response



model_dir = "/mnt/ymj/GLM-4-main/THUDM/glm-4-9b-chat"
# lora_dir = "/mnt/data/six/ymj/biaozhu/task6/checkpoint-600"

# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)

# 加载训练好的Lora模型
# model = PeftModel.from_pretrained(model, model_id=lora_dir)

# 将模型设置为评估模式
model.eval()

# 初始化列表来存储JSON数据
json_list = []

# 逐行处理数据并将结果添加到列表中


before_row = None
for index, row in tqdm(data.iterrows(), total=len(data)):
    #------------------推理---------------------------
    input_text = row['地区']  # 使用地区作为唯一标识符
    test_texts = {
        "instruction": "你是一个专业的数据分析师，请根据以下地区的各项性能指标数据，分析该地区是否存在异常情况。请按照以下格式分析每个指标：\n\n1. CPU使用情况分析：\n   - 最大后台CPU使用率分析：\n   - 平均后台CPU使用率分析：\n   - 最大峰值CPU使用率分析：\n\n2. 后台时间占比分析：\n   - 最大后台时间占比分析：\n   - 平均后台时间占比分析：\n\n3. 使用时长分析：\n   - 平均前台使用时长分析：\n\n4. 卸载情况分析：\n   - 最高卸载率分析：\n   - 平均卸载率分析：\n   - 短期卸载App占比分析：\n\n5. 综合分析：\n   - 异常分数解读：\n   - 与系统判定的对比分析：\n   - 最终结论：",
        "input": f"地区数据：\n\n地区名称：{row['地区']}\n\n1. CPU使用情况：\n最大后台CPU使用率：{row['区域_最大后台CPU使用率']}%\n平均后台CPU使用率：{row['区域_平均后台CPU使用率']}%\n最大峰值CPU使用率：{row['区域_最大峰值CPU使用率']}%\n\n2. 后台时间情况：\n最大后台时间占比：{row['区域_最大后台时间占比']}\n平均后台时间占比：{row['区域_平均后台时间占比']}\n\n3. 使用时长情况：\n平均前台使用时长：{row['区域_平均前台使用时长']}分钟\n\n4. 卸载情况：\n最高卸载率：{row['区域_最高卸载率App']}\n平均卸载率：{row['区域_平均卸载率']}\n短期卸载App占比：{row['区域_短期卸载App占比']}\n\n5. 异常情况：\n异常分数：{row['异常分数']}\n系统判定结果：{row['是否异常']}"
    }
    instruction = test_texts['instruction']
    input_value = test_texts['input']

    if before_row is None or input_text != before_row:
        messages_1 = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_value}
        ]

        response = predict(messages_1, model, tokenizer)  # 推理结果喂response
        response = response.strip()
        print(response)
        response = clean_and_merge_response(response)  # 清理并合并可能的多重列表

    message = {
        "地区": row['地区'],
        "异常分数": row['异常分数'],
        "是否异常": row['是否异常'],
        "分析结果": response
    }
    json_list.append(message)
    before_row = input_text  # 暂存上一组的文本

# 将数据转换为DataFrame并保存为CSV文件
output_df = pd.DataFrame(json_list)
output_path = '/mnt/ymj/vivo/地区异常/大模型推理结果/result.csv'
output_df.to_csv(output_path, index=False, encoding='utf-8')

# 同时保存一份JSON格式的结果
json_path = '/mnt/ymj/vivo/地区异常/大模型推理结果/result.json'
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_list, json_file, ensure_ascii=False, indent=4)

print(f'JSON文件已保存到: {json_path}')
