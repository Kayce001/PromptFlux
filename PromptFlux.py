import requests
from PIL import Image
from io import BytesIO
import gradio as gr
import time
import logging

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 直接在代码中赋值 API 密钥和 URL
api_key = "your-api-key"  # 你的 API 密钥
llm_url = "https://api.siliconflow.cn/v1/chat/completions"  # 语言模型 API 地址
image_gen_url = "https://api.siliconflow.cn/v1/images/generations"  # 图像生成 API 地址

# 提示词生成模板
template = """
Act as a stable diffusion Prompt Generator:
"I hope you can serve as a prompt generator, creating high-quality prompts based on user inputs for artificial intelligence programs. Your job is to provide detailed and creative descriptions that will inspire unique and interesting images from the AI. Keep in mind that the AI is capable of understanding a wide range of language and can interpret abstract concepts, so feel free to be as imaginative and descriptive as possible. Here is an example prompt: 'A field of wildflowers stretches out as far as the eye can see, each one a different color and shape. In the distance, a massive tree towers over the landscape, its branches reaching up to the sky like tentacles.'"
Please refine the following image generation prompt in English:
"""

def refine_prompt(user_prompt):
    """
    使用语言模型优化用户输入的提示词。
    """
    llm_payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct-128K",  # 更新后的模型名称
        "messages": [
            {
                "role": "user",
                "content": f"{template} {user_prompt}"
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": ["null"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }
    llm_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        logging.debug(f"发送给语言模型的负载: {llm_payload}")
        response = requests.request("POST", llm_url, json=llm_payload, headers=llm_headers, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        logging.debug(f"Refine Prompt Response: {response_json}")
        
        # 根据实际响应格式调整解析逻辑
        if "choices" in response_json and len(response_json["choices"]) > 0:
            refined_prompt = response_json["choices"][0].get("message", {}).get("content", "No content in message.")
        else:
            refined_prompt = "No refined prompt returned."
        return refined_prompt
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        return f"HTTP Request failed: {e}"
    except (ValueError, KeyError, IndexError) as e:
        logging.error(f"JSON Decode Error or Unexpected Response Format: {e}")
        return f"JSON Decode Error or Unexpected Response Format: {e}"

def generate_image(prompt, negative_prompt):
    """
    使用优化后的提示词生成图像。
    """
    payload = {
        "model": "black-forest-labs/FLUX.1-dev",  # 请根据实际情况确认模型名称
        "prompt": prompt,
        "negative_prompt": negative_prompt,  # 移除 "Negative prompt: " 前缀
        "image_size": "1024x1024",
        "batch_size": 1,
        "seed": 1234567890,  # 使用较小的种子值，确保在API允许范围内
        "num_inference_steps": 50,  # 增加步数以提高图像质量
        "guidance_scale": 7.5,
        "prompt_enhancement": False
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    max_retries = 5
    for i in range(max_retries):
        try:
            logging.debug(f"发送给图像生成API的负载: {payload}")
            response = requests.request("POST", image_gen_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            logging.debug(f"Generate Image Response: {response_json}")

            if "data" in response_json and len(response_json["data"]) > 0:
                image_url = response_json["data"][0].get("url")
                logging.debug(f"Image URL: {image_url}")
                if image_url:
                    image_response = requests.get(image_url, timeout=60)
                    image_response.raise_for_status()
                    image = Image.open(BytesIO(image_response.content))
                    return image
                else:
                    return "No image URL found in the response"
            else:
                return "No image data found in the response"
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {i+1} failed: {e}")
            time.sleep(2 ** i)  # 指数退避
    return "Failed to generate image after multiple attempts"

def integrated_generate_image(user_prompt):
    """
    集成优化提示词和生成图像的功能。
    """
    # 定义负面提示
    negative_prompt = (
        "Blurry, low quality, poor detail, unrealistic proportions, distortion, deformation, out of focus, anatomical errors, unnatural lighting, "
        "over saturation, grainy, pixelated, messy background, lack of detail, dull colors, flat, cartoony, overexposed, underexposed, "
        "poor hand details, twisted fingers, unnatural hand positioning, anatomical errors on hands, inharmonious hand shadows, "
        "blurred hand lines, disproportionate hand size, lacking texture on hand skin, incorrect number of fingers, incorrect drawn thumb."
    )

    # 先优化提示词
    refined_prompt = refine_prompt(user_prompt) 
    logging.debug(f"Optimized Prompt: {refined_prompt}")
    if "Error" in refined_prompt or "No refined prompt" in refined_prompt:
        return refined_prompt  # 返回错误信息

    # 使用优化后的提示词生成图像
    image = generate_image(refined_prompt, negative_prompt)
    if isinstance(image, Image.Image):
        return image
    else:
        return image  # 返回错误信息

# 创建 Gradio 界面
iface = gr.Interface(
    fn=integrated_generate_image,
    inputs=gr.Textbox(lines=2, placeholder="请输入图像描述..."),
    outputs=gr.Image(type="pil"),
    title="图像生成器",
    description="输入描述，生成图像",
    analytics_enabled=False  # 禁用分析
)

# 运行 Gradio 应用
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
