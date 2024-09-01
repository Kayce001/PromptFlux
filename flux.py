import requests
from cryptography.fernet import Fernet
from PIL import Image
from io import BytesIO
import re
import gradio as gr
import time

# 读取加密密钥
key_file_path = "secret.key"  # 确保路径正确
with open(key_file_path, "rb") as key_file:
    key = key_file.read()
cipher_suite = Fernet(key)

# 解密API密钥
encrypted_api_key = "gAAAAABmjKwb8aqffrV-moNQ6BEJEknwaY2301n5Nu0cNRFFgB98T0Z6shkHFA_2DM0PfiqpTcX5keE4WR0QEz7s3vC9JIrTVj53qdQibrxQlu7LbKPq9lzSM-ykL5V_gWQAVPXh9mlKYNclaQzAKct_xdGstIaf6Q=="

try:
    api_key = cipher_suite.decrypt(encrypted_api_key.encode()).decode()
except Exception as e:
    raise ValueError(f"Decryption failed: {e}")

# 模板字符串，用于激发用户提供详尽的描述
template = """
Act as a stable diffusion Prompt Generator:
"I hope you can serve as a prompt generator, creating high-quality prompts based on user inputs for artificial intelligence programs. Your job is to provide detailed and creative descriptions that will inspire unique and interesting images from the AI. Keep in mind that the AI is capable of understanding a wide range of language and can interpret abstract concepts, so feel free to be as imaginative and descriptive as possible,Here is an example prompt: "A field of wildflowers stretches out as far as the eye can see, each one a different color and shape. In the distance, a massive tree towers over the landscape, its branches reaching up to the sky like tentacles.".
"
Please refine the following image generation prompt in english:
""" 

# 使用LLM优化提示词
def refine_prompt(input_prompt):
    encrypted_llm_url = "gAAAAABmjKwbgAOc1v48A1v0mDAGpYzOyZu4fJm2u4vIDgMHHAHuEWz521Q2vnlToWO5dpc781hkwomCiW0d16tkJXxp-32qIr77E3jwqYw-NouLxiuXl_KwBkLLNgJ97xPvQN7N52N1"
    llm_url = cipher_suite.decrypt(encrypted_llm_url.encode()).decode()
    llm_payload = {
        "model": "Qwen/Qwen2-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": f"{template} {input_prompt}"
            }
        ]
    }
    llm_headers = {
        "Authorization": f"Bearer {api_key}",  # 确保从安全位置获取api_key
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(llm_url, json=llm_payload, headers=llm_headers)
        response.raise_for_status()  # 会抛出HTTP错误状态码的异常
        refined_prompt = response.json().get('choices')[0]['message']['content'] if response.json().get('choices') else "No refined prompt returned."
        return refined_prompt
    except requests.exceptions.RequestException as e:
        return f"HTTP Request failed: {e}"
    except ValueError as e:
        return f"JSON Decode Error: {e}"
    
def generate_image(prompt):
    url = "https://api.siliconflow.cn/v1/black-forest-labs/FLUX.1-schnell/text-to-image"  # 使用新的 URL
    payload = {
        "prompt": prompt,
        "image_size": "1024x1024",
        "batch_size": 1,
        "num_inference_steps": 25,
        "guidance_scale": 4.5
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }

    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            match = re.search(r'"url":"(https://[^"]+)"', response.text)
            if match:
                image_url = match.group(1)
                # 下载并返回图像
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                image = Image.open(BytesIO(image_response.content))
                return image
            else:
                return "No image URL found in the response"
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(2 ** i)  # Exponential backoff
    return "Failed to generate image after multiple attempts"


# 生成图像
def integrated_generate_image(input_prompt):
    # 先优化提示词
    negative_prompt = (
        "Negative prompt: Blurry, low quality, poor detail, unrealistic proportions, "
        "distortion, deformation, out of focus, anatomical errors, unnatural lighting, "
        "over saturation, grainy, pixelated, messy background, lack of detail, dull colors, "
        "flat, cartoony, overexposed, underexposed, poor hand details, twisted fingers, "
        "unnatural hand positioning, anatomical errors on hands, inharmonious hand shadows, "
        "blurred hand lines, disproportionate hand size, lacking texture on hand skin, incorrect number of fingers,incorrect drawn thumb.Please strictly follow the negative prompt instructions, ensuring that none of the mentioned elements are present."
    )

    # 先优化提示词
    refined_prompt = refine_prompt(input_prompt) 
    print("Optimized Prompt:", refined_prompt)
    if "Error" in refined_prompt:
        return refined_prompt  # 如果有错误，返回错误信息

    # 使用优化后的提示词生成图像
    return generate_image(refined_prompt)

# 禁用分析
gr.Interface.analytics_enabled = False

# 使用 Gradio 创建前端界面
iface = gr.Interface(
    fn=integrated_generate_image,  # 更新为新的集成函数
    inputs="text",
    outputs="image",
    title="Image Generation",
    description="输入描述，生成图像"
)

# 运行 Gradio 应用
iface.launch(server_name="0.0.0.0", server_port=7860)
