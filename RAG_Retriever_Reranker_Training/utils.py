from typing import List
import re
    
def get_inference_system_prompt() -> str:
    """get system prompt for generation"""
    prompt = """You are a helpful question-answering assistant. Extract direct answers from the provided passages.

Guidelines:
1. Provide direct, concise answers using information from the context
2. If the context contains relevant information, answer even if not perfectly complete
3. You may reasonably infer from the context when answering
4. Only use "CANNOTANSWER" when there is truly zero relevant information

Examples:
Q: "How badly was Marco Simoncelli hurt?"
Context: "Later, at a press conference involving members of the MotoGP Race Direction, Medical Director Michele Macchiagodena said that Simoncelli had sustained "a very serious trauma to the head, to the neck and the chest", and was administered CPR for 45 minutes. His body was flown home to Italy, accompanied by his father Paolo, his fiancee Kate Fretti, and Valentino Rossi."
Good: "a very serious trauma to the head, to the neck and the chest", and was administered CPR for 45 minutes."
Bad: "CANNOTANSWER" (context has relevant info)

Q: "what was mccarthy's coaching style like?"
Context: "While managing, McCarthy utilized a low-key approach, never going to the mound to remove a pitcher or arguing with an umpire except on a point of the rules, preferring to stay at his seat in the center of the dugout. He also declined to wear a numbered uniform with the Yankees and Red Sox."
Good: "utilized a low-key approach, never going to the mound to remove a pitcher or arguing with an umpire except on a point of the rules,"
Bad: "CANNOTANSWER" (context has relevant info)

Q: "What is the CEO's favorite color?"
Context: "The company has 500 employees..."
Good: "CANNOTANSWER" (no relevant info)"""
    return prompt

def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
    """Create user prompt with clear structure for better answer extraction"""
    # Format context passages with clear numbering
    formatted_contexts = ""
    for idx, context in enumerate(context_list, 1):
        context_text = context.strip()
        # Increased context length to give model more information (TOP_M=3, so limited context)
        if len(context_text) > 400:
            context_text = context_text[:400] + "..."
        formatted_contexts += f"[{idx}] {context_text}\n\n"
    
    prompt = f"""Question: {query}

Relevant passages:
{formatted_contexts}
Answer the question directly using the information above. If you can reasonably infer the answer from the passages, provide it. Only respond "CANNOTANSWER" if there is truly no relevant information.

Answer:"""
    return prompt

def parse_generated_answer(pred_ans: str) -> str:
    """Extract the actual answer from the model's generated text.
    
    The input pred_ans contains the FULL decoded sequence from tokenizer.batch_decode(outs).
    With Qwen models and skip_special_tokens=True, the format is typically:
    system\n{system_message}\nuser\n{user_message}\nassistant\n{generated_answer}
    
    We need to extract ONLY the assistant's answer part.
    """
    
    # CRITICAL: Remove Qwen3's thinking tags first
    # Qwen3 sometimes generates <think>...</think> even with enable_thinking=False
    pred_ans = re.sub(r'<think>.*?</think>', '', pred_ans, flags=re.DOTALL)
    pred_ans = re.sub(r'<think>.*$', '', pred_ans, flags=re.DOTALL)  # Handle unclosed tags
    
    # Strategy 1: Find content after "assistant" role marker
    # This works when skip_special_tokens=True removes <|im_start|> and <|im_end|>
    patterns = [
        # Pattern for when special tokens are removed: "...assistant\n{answer}"
        r'assistant\s*\n\s*(.+?)(?:\n|$)',
        r'assistant\s+(.+?)(?:\n|$)',
        
        # Pattern for when they're still there: "<|im_start|>assistant\n{answer}<|im_end|>"
        r'<\|im_start\|>\s*assistant\s*\n\s*(.+?)(?:<\|im_end\|>|$)',
        
        # Pattern looking for "Answer:" or "Answer (extract from context above):"
        r'Answer\s*(?:\([^)]+\))?\s*:\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, pred_ans, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            
            # Clean up any remaining special tokens
            answer = re.sub(r'<\|im_end\|>.*$', '', answer)
            answer = re.sub(r'<\|.*?\|>', '', answer)
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'</?think>', '', answer)
            
            # Take only the first line (avoid multi-line explanations)
            lines = [line.strip() for line in answer.split('\n') if line.strip()]
            if lines:
                answer = lines[0]
            
            # Final cleanup
            answer = answer.strip()
            
            # Reject if it's just a role name or empty
            if answer and len(answer) > 0 and answer.lower() not in ['assistant', 'system', 'user']:
                # Normalize common "cannot answer" variants to canonical token
                # Remove punctuation but keep spaces for pattern matching
                norm = re.sub(r"[^a-zA-Z0-9 ]", "", answer).strip().lower()
                cannot_patterns = [
                    r"^cannot ?answer$",
                    r"^cannotanswer$",
                    r"^cant answer$",        # "can't" → "cant" after punctuation removal
                    r"^no answer$",
                    r"^i do not know$",
                    r"^i dont know$",        # "don't" → "dont" after punctuation removal  
                    r"^unable to answer$",
                    r"^insufficient information$",
                    r"^cannot say$",
                    r"^cannot respond$",
                    r"^cannot reply$",
                ]
                for p in cannot_patterns:
                    if re.match(p, norm):
                        return "CANNOTANSWER"
                return answer
    
    # Strategy 2: If no "assistant" marker found, look for content after "Answer:"
    # This handles cases where the prompt format might vary
    answer_match = re.search(r'(?:^|\n)([^:\n]+)$', pred_ans.strip())
    if answer_match:
        answer = answer_match.group(1).strip()
        # Remove thinking tags
        answer = re.sub(r'</?think>', '', answer)
        # Make sure it's not part of the context or question or a role name
        if (answer and len(answer) > 3 and 
            answer.lower() not in ['assistant', 'system', 'user'] and
            not any(x in answer.lower() for x in 
                ['question:', 'context:', 'passage', 'based strictly', 'answer only', '<think>'])):
            return answer
    
    # Strategy 3: Last resort - split by double newlines and take the last substantial part
    parts = [p.strip() for p in pred_ans.split('\n\n') if p.strip()]
    if parts:
        for part in reversed(parts):
            # Skip parts that look like system/user prompts or thinking tags
            if not any(x in part.lower() for x in ['system', 'user', 'question:', 'context:', 'passage', '<think>']):
                # Remove role markers and thinking tags
                part = re.sub(r'^(assistant|system|user)\s*', '', part, flags=re.IGNORECASE)
                part = re.sub(r'</?think>', '', part)
                part = part.strip()
                # Reject if it's just a role name
                if part and len(part) > 3 and part.lower() not in ['assistant', 'system', 'user']:
                    # Take first line
                    lines = [line.strip() for line in part.split('\n') if line.strip()]
                    if lines:
                        first_line = lines[0]
                        # Final check - not a role name
                        if first_line.lower() not in ['assistant', 'system', 'user']:
                            return first_line
    
    # If all else fails, return empty string
    return ""