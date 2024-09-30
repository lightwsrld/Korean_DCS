import transformers
import torch

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. [요약문]을 보고 output template에 맞게 구체적으로 작성해주세요.'''
instruction = '''
[subject_keyword]
"해외여행"
[output template]
두 화자는 이 대화에서 [요약]에 대해 말했습니다.
[Dialogue]

[요약문]
SD2000001은 여행을 좋아하여 국내, 해외 여행을 많이 다녔다고 말했습니다. 특히 기억에 남는 여행지로 스페인 마드리드의 톨레도를 소개했습니다. 그 중 화려하게 꾸며진 대성당과 디저트가 인상적이었다고 이야기했습니다. SD2000002는 대학교에 진학한 후 해외여행을 자주 다녔고, 스페인과 포루투갈이 가장 기억에 남는 여행지라고 말했습니다. 그리고 톨레도도 다녀왔지만 날씨가 더워서 제대로 구경하지 못했다는 경험을 이야기했습니다.
'''

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print(outputs[0]["generated_text"][len(prompt):])