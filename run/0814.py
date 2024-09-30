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

PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. 위 subject_keyword에 대한 대화를 output template에 맞게 "3문장 이내로" 요약해주세요.'''
instruction = '''
[subject_keyword]
"해외여행"
[output template]
[화자 1]은 [화자 1의 요점]이라고 말했습니다. [추가 설명] 또한 [관련 발언 요약]라고 말했습니다.
[Dialogue]
"speaker": "SD2000001" 
"utterance": 저는 여행 다니는 것을 굉장히 좋아하는데요. 그래가지고 스페인이나 뭐 영국 유럽 아니면 국내에서도 뭐 강릉이나 전주 같은 데를 많이 다녔는데 혹시 여행 다니는 거 좋아하시나요?
어~ 네. 저도 우연히 스페인과 포르투갈을 다녀왔었었습니다.
어~ 저는 스페인 중에서도 마드리드에 근교에 있었던 톨레도라는 지역이 굉장히 좋았는데요. 그 톨레도에서 특히 기억에 남았던 거는 거기에 대성당이 있는데 그 성당이 엄청 화려하더라고요. 그래서 거기를 꾸며논 거를 보면은 금을 엄청 많이 사용해가지고 되게 빤짝빤짝하고 좀 성당은 보통 좀 소박하다라는 인식이 있었는데 아~ 이렇게 화려한 성당도 있구나라는 거를 새롭게 알게 됐었습니다. 어~ 또 톨레도에 지역 음식도 같이 먹었었는데 아~ 이름은 지금 잘 생각이 나지는 않지만 굉장히 달달했던 그런 디저트 종류였는데 그~ 디저트도 먹고 그다음에 천천히 걸어 다니면서 주변 풍경도 보고 근교 여행만의 약간 소박한 맛이 있었다고 생각을 합니다. 어~ 또 물론 마드리드도 굉장히 좋았는데 유럽 여행을 많이 가셨다고 해서 혹시 톨레도도 가본 적이 있나요?
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