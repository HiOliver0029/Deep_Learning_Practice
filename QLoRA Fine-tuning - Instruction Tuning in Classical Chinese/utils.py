from transformers import BitsAndBytesConfig
import torch

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,              # 是否用 8-bit 量化（如果 True，會以 8bit 取代原始 FP16/FP32）
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload for insufficient GPU memory
    )

# def get_bnb_config_minimal() -> BitsAndBytesConfig:
#     '''Get minimal BitsAndBytesConfig for low GPU memory.'''
#     return BitsAndBytesConfig(
#         load_in_4bit=True,
#         load_in_8bit=False,              # 是否用 8-bit 量化（如果 True，會以 8bit 取代原始 FP16/FP32）
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,  # 是否使用 double quantization（二次量化，進一步壓縮記憶體用量）
#         bnb_4bit_compute_dtype=torch.float16,  # Use float16 instead of bfloat16
#         llm_int8_enable_fp32_cpu_offload=True,
#     )

def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是古文的專家，負責轉換古代文言文與現代白話文。接下來是你跟用戶的對話，你要對用戶的問題提供簡潔、有用、精確的轉換。USER:{instruction} ASSISTANT:"

def get_prompt_few_shot(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    prompt = "你是古文的專家，負責轉換古代文言文與現代白話文。接下來是你跟用戶的對話，你要對用戶的問題提供簡潔、有用、精確的轉換。以下為幾個正確翻譯的範例 \n"
    prompt +=   """USER:{ "翻譯成文言文:雅裏惱怒地說: 從前在福山田獵時，你誣陷獵官，現在又說這種話。答案:"，} ASSISTANT:{"雅裏怒曰: 昔畋於福山，卿誣獵官，今復有此言。"} \n
                USER:{"沒過十天，鮑泉果然被拘捕。幫我把這句話翻譯成文言文"} ASSISTANT:{"後未旬，果見囚執。"} \n  
                USER:{"辛未，命吳堅為左丞相兼樞密使，常楙參知政事。把這句話翻譯成現代文。"} ASSISTANT:{"初五，命令吳堅為左承相兼樞密使，常增為參知政事。"} \n
                USER:{"翻譯成現代文:士匄請見，弗內。答案:"} ASSISTANT:{"士匄請求進見，荀偃不接見。"} \n
                USER:{"將下麵句子翻譯成文言文:今天令在必行，不得有任何阻攔。"} ASSISTANT:{"自今令在必行，毋有所遏。"} \n 
                USER:{"文言文翻譯:則其製衣，宜若生人之服。"} ASSISTANT:{"答案:那麼給它做衣服就應該像活人的衣服一樣。"} \n""" 
    prompt += f'\n接下來是你要翻譯的句子，根據前面的正確範例，進行翻譯\n USER: {instruction} ASSISTANT:'
    return prompt

# def get_prompt_few_shot(instruction: str, examples: list) -> str:
#     '''Format the instruction with few-shot examples as a prompt for LLM.'''
#     prompt = "你是中文系的專業教授，負責轉換古代文言文與現代白話文。接下來是你跟用戶的對話，你要對用戶的問題提供有用、正確的轉換。\n\n"
    
#     # Add examples
#     for example in examples:
#         prompt += f"USER: {example['instruction']} ASSISTANT: {example['output']}\n\n"
    
#     # Add current instruction
#     prompt += f"USER: {instruction} ASSISTANT:"
    
#     return prompt

def get_prompt_v2_concise(instruction: str) -> str:
    '''更簡潔的 prompt 格式'''
    return f"古文專家，請完成：USER:{instruction}\n答："

def get_prompt_v3_specific(instruction: str) -> str:
    '''針對翻譯任務優化的 prompt'''
    if "翻譯" in instruction or "translation" in instruction.lower():
        return f"你是專業古文翻譯專家。請準確翻譯以下內容。\n\n{instruction}\n\n翻譯結果："
    else:
        return f"你是古文專家。請準確完成以下任務。\n\n{instruction}\n\nASSISTANT:"

def get_prompt_v4_structured(instruction: str) -> str:
    '''結構化的 prompt，添加格式指導'''
    return f"""你是古文專家。請按以下要求完成任務：

任務：{instruction}

要求：
- 回答簡潔準確
- 保持語言風格一致
- 不要添加無關內容

ASSISTANT:"""

def get_prompt_v5_zero_shot(instruction: str) -> str:
    '''零樣本優化 prompt'''
    return f"# 古文專家\n\n請完成：{instruction}\n\n## 回答\n"



def get_prompt_enhanced(instruction: str) -> str:
    '''Enhanced prompt format optimized for classical Chinese tasks.'''
    return f"""你是專業的古典文學教授，精通古代文言文與現代白話文的互相轉換。你的任務是準確理解用戶的指令，並提供精確、有用的回答。

            請注意：
            - 準確理解古文的語法結構和詞彙含義
            - 保持翻譯的文學性和準確性
            - 根據上下文提供最合適的現代表達

            USER: {instruction} ASSISTANT:"""

def get_prompt_concise(instruction: str) -> str:
    '''Concise prompt format for better efficiency.'''
    return f"你是古文專家，負責文言文與白話文轉換。USER: {instruction} ASSISTANT:"

def get_prompt_roleplay(instruction: str) -> str:
    '''Role-playing prompt format for better context understanding.'''
    return f"""你是一位博學的古典文學教授，學生向你請教古文相關問題。你需要：
1. 準確理解學生的問題
2. 提供清楚、正確的解答
3. 必要時給出詳細的解釋

學生問：{instruction}
教授答："""