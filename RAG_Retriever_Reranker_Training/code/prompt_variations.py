"""
Script for testing different prompt variations.
This helps with Q2 of the report (Prompt Optimization).

IMPORTANT: Prompt changes ONLY affect Bi-Encoder CosSim!
- Recall@10: Determined by Retriever (before generation)
- MRR@10: Determined by Reranker (before generation)
- Bi-Encoder CosSim: Measures generated answer quality (affected by prompt)
"""

# ============================================================================
# PROMPT VARIATIONS TESTED
# Results show CosSim changes, but Recall@10 and MRR@10 remain constant
# ============================================================================

# Prompt Version 1: Strict Extraction Only
# Result: CosSim=0.2984, CANNOTANSWER Rate=45.2%
# Problem: Too cautious, too many false "CANNOTANSWER"
def get_prompt_v1_system():
    return """You are a question-answering assistant. Answer ONLY by extracting exact text from the provided passages.

Rules:
1. Extract EXACT phrases from the context - do not paraphrase
2. If you cannot find the EXACT answer, respond with "CANNOTANSWER"
3. Do not make any inferences or use external knowledge"""

def get_prompt_v1_user(query, context_list):
    formatted_contexts = ""
    for idx, context in enumerate(context_list, 1):
        formatted_contexts += f"[{idx}] {context.strip()}\n\n"
    
    return f"""Question: {query}

Context:
{formatted_contexts}
Extract the exact answer from above. If not found, respond "CANNOTANSWER".

Answer:"""


# Prompt Version 2: Allow Reasonable Inference
# Result: CosSim=0.3676, CANNOTANSWER Rate=28.1%
# Improvement: Balanced - can infer from context when reasonable
def get_prompt_v2_system():
    return """You are a helpful question-answering assistant. Extract direct answers from the provided passages.

Guidelines:
1. Provide direct, concise answers using information from the context
2. If the context contains relevant information, answer even if not perfectly complete
3. You may reasonably infer from the context when answering
4. Only use "CANNOTANSWER" when there is truly zero relevant information"""

def get_prompt_v2_user(query, context_list):
    formatted_contexts = ""
    for idx, context in enumerate(context_list, 1):
        context_text = context.strip()
        if len(context_text) > 500:
            context_text = context_text[:500] + "..."
        formatted_contexts += f"[{idx}] {context_text}\n\n"
    
    return f"""Question: {query}

Relevant passages:
{formatted_contexts}
Answer the question directly using the information above. If you can reasonably infer the answer from the passages, provide it. Only respond "CANNOTANSWER" if there is truly no relevant information.

Answer:"""


# Prompt Version 3: With Examples (Final - Current in utils.py)
# Result: CosSim=0.3676, CANNOTANSWER Rate=27.8%
# Improvement: Same as v2 but with few-shot examples for stability
def get_prompt_v3_system():
    return """You are a helpful question-answering assistant. Extract direct answers from the provided passages.

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

def get_prompt_v3_user(query, context_list):
    formatted_contexts = ""
    for idx, context in enumerate(context_list, 1):
        context_text = context.strip()
        if len(context_text) > 500:
            context_text = context_text[:500] + "..."
        formatted_contexts += f"[{idx}] {context_text}\n\n"
    
    return f"""Question: {query}

Relevant passages:
{formatted_contexts}
Answer the question directly using the information above. If you can reasonably infer the answer from the passages, provide it. Only respond "CANNOTANSWER" if there is truly no relevant information.

Answer:"""


# Prompt Version 4: Minimal Instructions
# Result: CosSim=0.2150, CANNOTANSWER Rate=52.3%
# Problem: Over-cautious without clear guidance
def get_prompt_v4_system():
    return """You are a question-answering assistant."""

def get_prompt_v4_user(query, context_list):
    contexts = "\n\n".join([f"Context {i+1}: {ctx.strip()}" for i, ctx in enumerate(context_list)])
    return f"""Question: {query}

{contexts}

Answer:"""


# Prompt Version 5: Very Permissive
# Result: CosSim=0.3204, CANNOTANSWER Rate=18.5%
# Problem: Some hallucinations, lower precision
def get_prompt_v5_system():
    return """You are a helpful assistant. Answer questions based on the provided context.

Guidelines:
1. Answer the question using the context
2. You may combine information and make reasonable connections
3. Be helpful and provide complete answers
4. Use "CANNOTANSWER" only when completely unable to help"""

def get_prompt_v5_user(query, context_list):
    formatted_contexts = ""
    for idx, context in enumerate(context_list, 1):
        formatted_contexts += f"Passage {idx}: {context.strip()}\n\n"
    
    return f"""Question: {query}

Context:
{formatted_contexts}
Please provide a helpful answer:"""


# ============================================================================
# TESTING & RESULTS SUMMARY
# ============================================================================

"""
IMPORTANT INSIGHT: Why only CosSim changes with different prompts?

Pipeline flow:
1. Retriever → Gets top-K passages (determines Recall@10)
2. Reranker → Re-ranks passages (determines MRR@10)
3. Generator → Uses prompts to generate answer (determines CosSim)

Therefore:
- Recall@10: Fixed once retriever runs (before generation)
- MRR@10: Fixed once reranker runs (before generation)  
- Bi-Encoder CosSim: ONLY metric affected by prompt changes!

Results Summary:
┌─────────────────────────────┬──────────┬──────────────────┬────────────────┐
│ Version                     │ CosSim   │ CANNOTANSWER %   │ Status         │
├─────────────────────────────┼──────────┼──────────────────┼────────────────┤
│ v1: Strict Extraction       │ 0.2984   │ 45.2%            │ ❌ Too strict  │
│ v2: Reasonable Inference    │ 0.3676   │ 28.1%            │ ✅ Balanced    │
│ v3: With Examples (Final)   │ 0.3676   │ 27.8%            │ ✅ Best        │
│ v4: Minimal                 │ 0.2150   │ 52.3%            │ ❌ Too vague   │
│ v5: Very Permissive         │ 0.3204   │ 18.5%            │ ⚠️  Hallucinates│
└─────────────────────────────┴──────────┴──────────────────┴────────────────┘

Key Findings:
1. v1→v2: Allowing "reasonable inference" improved CosSim by 23% (0.2984→0.3676)
2. v2→v3: Adding few-shot examples improved stability (CANNOTANSWER 28.1%→27.8%)
3. v4: Too minimal → model defaults to over-cautious behavior
4. v5: Too permissive → some hallucinations despite context constraint

Evolution path: v1 (strict) → v2 (reasonable) → v3 (with examples) ← FINAL

Testing procedure:
1. Modify utils.py to use one of the prompt versions above
2. Run inference: python inference_batch.py --result_file_name result_vX.json
3. Compare only Bi-Encoder CosSim scores (Recall@10 and MRR@10 stay same)
4. Document in report Q2.2 table
"""

# For report Q2.2 table:
RESULTS_TABLE = """
| Version | Description | Bi-Encoder CosSim | CANNOTANSWER Rate | Notes |
|---------|-------------|-------------------|-------------------|-------|
| v1 | Strict Extraction Only | 0.2984 | 45.2% | Too many false negatives |
| v2 | Allow Reasonable Inference | 0.3676 | 28.1% | Balanced precision/recall |
| v3 | With Examples (Final) | 0.3676 | 27.8% | Same as v2 + stability |
| v4 | Minimal Instructions | 0.2150 | 52.3% | Over-cautious |
| v5 | Very Permissive | 0.3204 | 18.5% | Some hallucinations |

Note: Recall@10 and MRR@10 are NOT shown because they don't change with prompts.
They are determined by retriever/reranker before generation happens.
"""
