import json

import nltk
from konlpy.tag import Okt
import torch
from torch.utils.data import Dataset

def add_subject(data, top_n):
    # Ensure necessary NLTK data files are available
    nltk.download('punkt')

    # Load stopwords from the file and remove duplicates
    stop_words = None
    with open("stop_words.txt", 'r', encoding='utf-8') as file:
        stop_words = set(file.read().split())

    okt = Okt()
    text = " ".join(utterance['utterance'] for conversation in data for utterance in conversation['input']['conversation'])

    # Calculate global word frequencies
    word_frequency = {}
    for noun in okt.nouns(text):
        word_frequency[noun] = word_frequency.get(noun, 0) + 1

    global_rank = {word: rank for rank, (word, _) in enumerate(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))}
    global_len = len(global_rank)

    # Process each conversation and calculate local word frequencies
    for conversation in data:
        text = " ".join(utterance['utterance'] for utterance in conversation['input']['conversation'])

        local_word_frequency = {}
        for noun in okt.nouns(text):
            local_word_frequency[noun] = local_word_frequency.get(noun, 0) + 1

        scored_keywords = []
        for word, freq in sorted(local_word_frequency.items(), key=lambda x: x[1], reverse=True):
            if word in stop_words:
                continue
            global_word_rank = global_rank.get(word, global_len)
            score = round(((global_word_rank + 1) / global_len) / ((local_word_frequency[word] + 1) / freq), 2)
            scored_keywords.append((word, score))

        scored_keywords = sorted(scored_keywords, key=lambda x: x[1], reverse=True)[:top_n]

        # Update the 'subject_keyword' field in the JSON data
        conversation['input'].setdefault('sub_subject_keyword', []).extend(word for word, _ in scored_keywords)
        conversation['input']['sub_subject_keyword'] = list(set(conversation['input']['sub_subject_keyword']))
        
    return data

class CustomDataset(Dataset):
    def __init__(self, fnames, tokenizer, pt=(0,1)):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        data = None
        for fname in fnames:
            if not data:
                with open(fname, "r") as f:
                    data = json.load(f)
            else:
                with open(fname, "r") as f:
                    data += json.load(f)
        
        data = data[int(len(data)*pt[0]): int(len(data)*pt[1])]
        data = add_subject(data, 10)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 3문장 이내로 요약해주세요. 요약에 참고할 키워드는 {', '.join(inp['sub_subject_keyword'])} 입니다. \n [Output Format] \n [화자]은 [화자의 요점]이라고 말했습니다. [추가 설명] 또한 [관련 발언 요약]라고 말했습니다. \n [Caution] \n  주어진 대화에서 화자가 이야기한 모든 내용이 꼭 들어가야합니다."
            chat = chat + "\n\n" + question

            return chat
        
        for example in data:
            PROMPT = f'''
당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.
주어진 대화 내용을 바탕으로 [Task]를 수행하세요.

[예시1]
[Conversation]
화자SD2000001: 저는 여행 다니는 것을 굉장히 좋아하는 데요. 그래가지고 스페인이나 뭐 영국 유럽 아니면 국내 에서도 뭐 강릉이나 전주 같은 데를 많이 다녔는데
화자SD2000001: 혹시 여행 다니는 거 좋아하시나요?
화자SD2000001: 어~ 네. 저도 우연히 스페인과 포르투갈 을 다녀왔었었습니다.
화자SD2000001: 어~ 저는 스페인 중에서도 마드리드에 근교에 있었던 톨레도라는 지역이 굉장히 좋았는데요. 그  톨레도에서 특히 기억에 남았던 거는 거기에 대성당이 있는데 그 성당이 엄청 화려하더라고요. 그래서 거기를 꾸 며논 거를 보면은 금을 엄청 많이 사용해가지고 되게 빤 짝빤짝하고 좀 성당은 보통 좀 소박하다라는 인식이 있었는데 아~ 이렇게 화려한 성당도 있구나라는 거를 새롭게 알게 됐었습니다.
화자SD2000001: 어~ 또 톨레도에 지역 음식도 같이 먹었 었는데 아~ 이름은 지금 잘 생각이 나지는 않지만 굉장히 달달했던 그런 디저트 종류였는데 그~ 디저트도 먹고 그다음에 천천히 걸어 다니면서 주변 풍경도 보고 근교 여 행만의 약간 소박한 맛이 있었다고 생각을 합니다.
화자SD2000001: 어~ 또 물론 마드리드도 굉장히 좋았는데 유럽 여행을 많이 가셨다고 해서 혹시 톨레도도 가본 적이 있나요?

[Question]
위 해외여행 주제에 대한 대화를 3문장 이내로 요약해주세요. 요약에 참고할 키워드는 달달, 근교, 톨레도, 빤짝, 데마, 대성당, 고대, 마드리드, 포르투갈, 관광버스 입니다.
[Output Format] 
[화자]은 [화자의 요점]이라고 말했습니다. [추가 설명] 또한 [관련 발언 요약]라고 말했습니다. 
[Caution] 
주어진 대화에서 화자가 이야기한 모든 내용이 꼭 들어가야합니다.

[Answer]
이 대화에서 화자들은 좋았던 여행지와 기억나는 주요 명소에 대해 이야기했습니다. SD2000001은 여행을 좋아하여 국내, 해외 여행을 많이 다녔다고 말했습니다. 특히 기억에 남는 여행지로 스페인 마드리드의 톨레도를 소개했습니다. 그 중 화려하게 꾸며진 대성당과 디저트가 인상적이었다고 이야기했습니다. SD2000002는 대학교에 진학한 후 해외여행을 자주 다녔고, 스페인과 포루투갈이 가장 기억에 남는 여행지라고 말했습니다. 그리고 톨레도도 다녀왔지만 날씨가 더워서 제대로 구경하지 못했다는 경험을 이야기했습니다.

[예시2]
[Conversation]
화자SD2000005: name1 씨는 평소에 즐겨먹는 음식이 있나요?
화자SD2000005: 저는 굳이 야채파 고기파 나누자면 고기 파인 거 같습니다.
화자SD2000005: 일단 야채는 좀 야채만 먹으면 되게 그  먹은 순간에는 되게 포만감이 있을 수 있는데 금방 꺼지 기도 해서 금방 허기가 지 져서 별로 안 좋아하고요. 어~ 고기는 좀만 먹어도 금방 꺼지지 않으니까 되게 야채파 고기파를 나누자면 고기파인 거 같습니다.
화자SD2000005: 고기도 되게 종류가 많은데 그 중에서도 그냥 돼지나 소보다는 그냥 닭고기가 좀 더 선호하고 되 게 주위에서 흔하게 가장 먹을 수 있어서 더 좋은 거 같 아요.
화자SD2000005: name1 씨는 간단하게 때울 수 있는 음식 으로는 주로 뭘 드시나요?
화자SD2000005: name1 씨는 불량식품도 혹시 드시는 거  있으신가요?

[Question]
위 음식, 불량식품 주제에 대한 대화를 3문장 이내로 요약해주세요. 요약에 참고할 키워드는 고기파이, 아폴로, 팟타이, 허기, 클리어, 뚝배기, 쥬스, 버거킹, 사탕, 제육 입니다.
[Output Format] 
[화자]은 [화자의 요점]이라고 말했습니다. [추가 설명] 또한 [관련 발언 요약]라고 말했습니다. 
[Caution] 
주어진 대화에서 화자가 이야기한 모든 내용이 꼭 들어가야합니다.

[Answer]
두 화자는 이 대화에서 즐겨 먹는 음식, 간단하게 먹을 수 있는 음식 등에 대해 말했습니다. SD2000005는 고기와 야채 중 먹고 난 후 포만감이 금방 꺼지지 않는 고기를 더 선호한다고 말했습니다. SD2000006은 찌개 종류를 좋아하고, 베트남 음식과 떡볶이도 좋아하여 많이 먹는다고 말했습니다. 그리고 시간이 없어 간단하게 먹어야 할 때는 편의점을 이용하며, 김밥이나 샌드위치 종류로 먹는다고 했습니다. 또 밥을 빨리 먹는 편이라서 코스요리가 아니라면 음식점을 이용한다고 말했습니다. 그리고 불량 식품 보다는 세계 과자점을 이용하는 편이라고 이야기했습니다.

'''
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target_text = example["output"]  # target 변수명을 target_text로 변경
            target = example["output"]
            if target != "":
                target += tokenizer.eos_token
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )