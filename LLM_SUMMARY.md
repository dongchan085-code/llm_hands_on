# AI 인증시험 LLM 상세 요약

이 문서는 `AI인증시험_출제포인트.pdf`와 `backup/llm_hands_on/pdf`의 LLM 강의 PDF, 그리고 관련 실습 노트북 내용을 바탕으로 시험 대비용으로 재정리한 상세 요약이다. 시험에서 특히 중요한 축은 “모델 구조를 말로 설명할 수 있는가”, “텐서 shape 흐름을 따라갈 수 있는가”, “학습 데이터와 loss가 어떻게 연결되는가”, “RAG와 온디바이스 최적화 개념을 구분할 수 있는가”이다.

## 1. 시험 출제 방향

### 1.1 LLM 영역 핵심 체크포인트

- 주요 layer의 의미, 동작 원리, 구성 방법, 활용 방법을 이해해야 한다.
- 데이터 입력부터 출력까지 Tensor 값과 shape의 주요 흐름을 설명하고 구성할 수 있어야 한다.
- 모델에서 어떤 내용을 학습하는지, 그 학습에 필요한 데이터 구성과 loss 계산을 이해해야 한다.
- 모델과 학습 또는 평가 의도에 맞게 데이터를 구성하는 방법을 알아야 한다.
- 코드 빈칸형 문제에서는 `Dataset`, `Attention`, `GPTModel`, `LoRA`, `Instruction/DPO collate`, `loss` 계산이 출제되기 쉽다.

### 1.2 RAG 영역 핵심 체크포인트

- RAG의 구성 요소와 주요 흐름을 이해해야 한다.
- retrieval 결과가 LLM generation 품질에 어떤 영향을 주는지 설명할 수 있어야 한다.
- `llama-index`, MCP 같은 RAG 관련 도구의 역할을 개념적으로 구분해야 한다.

### 1.3 온디바이스 최적화 핵심 체크포인트

- Quantization은 layer 단위에서 양자화/역양자화가 어떻게 이뤄지는지 이해해야 한다.
- Pruning은 가중치 또는 구조의 중요도를 판단하고 불필요한 연산을 제거하는 과정이다.
- Distillation은 teacher 모델의 예측값과 정답값을 student 모델 학습에 연결하는 방식이다.

### 1.4 시험 범위 안내 (◎ vs ⚠ 참고용)

본 문서는 강의 자료(커리큘럼 PDF) 전체 토픽을 다루지만, *시험 출제 포인트 PDF 키워드* 에 포함되지 않는 항목들은 모두 다음 배너로 명시했다.

```text
> ⚠ 시험 출제 포인트 외 — 참고용
```

해당 배너가 달린 섹션·문단은 강의에서는 다뤄지지만 26.5월 시험 출제 포인트 PDF 키워드에는 등장하지 않는다. 시험 직전에는 ◎ 영역(LLM 모델 원리·데이터 구성, RAG, On-device 3종)을 우선하라.

참고용으로 분류된 주요 토픽: **LoRA / QLoRA, DPO / RLHF / PPO, Optimizer 종류, Top-K · Temperature decoding, KV cache, GQA, MLA, MoE, Sliding Window Attention, Flash Attention, Gated DeltaNet**.

(Instruction fine-tuning, Classification fine-tuning 은 "학습/평가 의도에 맞는 데이터 구성" 안에 포함되는 ◎ 항목으로 본다.)

## 2. LLM 전체 흐름 한눈에 보기

### 2.1 학습 데이터에서 logits까지

```text
원본 텍스트
-> tokenizer.encode()
-> token ids [B, T]
-> token embedding [B, T, D]
-> positional embedding [T, D]
-> input embedding [B, T, D]
-> TransformerBlock x N
-> final LayerNorm
-> output head
-> logits [B, T, V]
-> Cross Entropy loss
```

- B: batch size
- T: sequence length 또는 context length
- D: embedding dimension
- V: vocabulary size

### 2.2 GPT 학습의 핵심

GPT는 현재까지의 토큰으로 다음 토큰을 예측한다. 입력과 정답은 같은 텍스트에서 만들지만, 정답은 입력보다 한 칸 뒤로 밀려 있다.

```python
input_chunk = token_ids[i : i + max_length]
target_chunk = token_ids[i + 1 : i + max_length + 1]
```

예를 들어 입력이 `[나는, 학교에, 간다]`라면 target은 `[학교에, 간다, ...]`가 된다. 모델은 각 위치에서 “다음 토큰”을 맞히도록 학습된다.

## 3. 수학 기초: Softmax, Sigmoid, Cross Entropy, KL Divergence

### 3.1 Softmax

Softmax는 여러 출력값, 즉 logits를 확률분포로 바꾼다.

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

- 모든 class 확률의 합은 1이다.
- 다중 분류에서 사용된다.
- 각 class 확률이 서로 독립적이라기보다 전체 class 사이에서 상대적으로 정해진다.
- logits 값의 차이가 클수록 softmax 결과는 더 확정적으로 된다.

### 3.2 Sigmoid와 Softmax의 관계

- Sigmoid는 하나의 값을 0과 1 사이 확률로 바꾼다.
- 이진 분류에서 자주 사용된다.
- class가 2개인 softmax는 sigmoid와 같은 형태로 해석할 수 있다.

시험 포인트:

- 다중 분류: Softmax
- 이진 분류: Sigmoid 또는 binary cross entropy
- GPT next-token prediction: vocabulary 전체에 대한 softmax

### 3.3 Cross Entropy

Cross Entropy는 정답 분포와 모델 예측 분포의 차이를 측정한다. LLM에서는 정답 token의 확률이 높아지도록 loss를 줄인다.

```text
CE = -log P(correct_token)
```

- 정답 token 확률이 1에 가까우면 loss가 작다.
- 정답 token 확률이 낮으면 loss가 크다.
- GPT 학습에서는 모든 위치의 정답 token 확률을 이용해 평균 loss를 계산한다.

### 3.4 KL Divergence

KL Divergence는 두 확률분포의 차이를 측정한다.

```text
D_KL(P || Q) = CrossEntropy(P, Q) - Entropy(P)
```

강의 PDF에서는 “모델이 실제 데이터 분포에서 얼마나 벗어났는가” 또는 “모델이 급격하게 바뀌면 안 된다”는 관점에서 등장한다. DPO나 RLHF 계열에서는 policy model이 reference model에서 너무 멀어지지 않게 하는 관점과 연결된다.

## 4. 벡터 공간과 임베딩

### 4.1 임베딩의 의미

임베딩은 단어, 토큰, 이미지, 범주형 데이터 등을 벡터 공간의 점으로 표현하는 것이다.

- 비슷한 의미의 단어는 벡터 공간에서 가까운 위치에 놓이도록 학습된다.
- 단어 간 유사도는 cosine similarity로 계산할 수 있다.
- `gensim` 실습에서는 `similarity`, `most_similar`, `doesnt_match` 같은 기능으로 의미적 관계를 확인한다.

### 4.2 Token Embedding

토큰 ID는 숫자일 뿐이므로 그대로는 의미를 담지 못한다. `nn.Embedding(vocab_size, emb_dim)`은 각 token id를 학습 가능한 벡터로 바꾼다.

```python
token_embedding_layer = torch.nn.Embedding(vocab_size, emb_dim)
token_embeddings = token_embedding_layer(input_ids)
```

shape:

```text
input_ids:         [B, T]
token_embeddings: [B, T, D]
```

시험 포인트 ★ — **`nn.Embedding` 의 두 인자**:

- 첫 번째 인자 = **`vocab_size`** (단어장 크기). 가능한 token id 의 개수.
- 두 번째 인자 = **`emb_dim` (= D)**. 한 토큰을 표현할 벡터의 차원.
- 내부 weight 행렬은 `[vocab_size, D]` shape 의 *학습 대상* 파라미터다. 한 행이 한 token 의 의미 벡터.
- 입력 `[B, T]` (정수 token id) → 출력 `[B, T, D]`. 즉 토큰 id 가 자기 행에 해당하는 벡터로 *조회 (lookup)* 된다.

### 4.3 Positional Embedding

Transformer는 RNN처럼 순서를 자연스럽게 처리하지 않으므로 위치 정보를 따로 넣어야 한다.

```python
pos_embedding_layer = torch.nn.Embedding(context_length, emb_dim)
pos_embeddings = pos_embedding_layer(torch.arange(seq_len))
input_embeddings = token_embeddings + pos_embeddings
```

shape:

```text
token_embeddings: [B, T, D]
pos_embeddings:   [T, D]
input_embeddings: [B, T, D]
```

`[T, D]` 위치 임베딩은 batch 차원에 브로드캐스팅되어 모든 샘플에 더해진다.

시험 포인트 ★ — **Positional Embedding 차원과 최종 합산**:

- 첫 번째 인자 = `context_length` (모델이 받을 수 있는 *최대 시퀀스 길이* `T_max`).
- 두 번째 인자 = `emb_dim` (= D). **token embedding 의 두 번째 인자와 같아야 한다.** 합산 (덧셈) 이 가능하려면 마지막 차원이 같아야 하기 때문이다.
- 입력으로는 위치 인덱스 `torch.arange(seq_len)` = `[0, 1, ..., T-1]` 을 통과시킨다. 결과 shape `[T, D]` 에는 batch 차원이 *없다*.
- 최종 합산은 **element-wise 덧셈** (곱셈이 아님). `[B, T, D] + [T, D]` 는 PyTorch broadcasting 으로 `[B, T, D]` 가 된다 — *같은 위치 임베딩이 batch 의 모든 샘플에 똑같이 더해진다*.

### 4.4 Sinusoidal Positional Encoding과 RoPE

Sinusoidal positional encoding:

- 삼각함수로 절대 위치 정보를 만든다.
- 낮은 차원은 주기가 짧아 세밀한 위치를 구분한다.
- 높은 차원은 주기가 길어 긴 문장에서도 위치 차이를 표현한다.

RoPE, Rotary Positional Embedding:

- 위치와 각도에 비례해 벡터를 회전시키는 방식이다.
- 상대적 거리 정보 보존에 유리하다.
- KV cache와 잘 맞아 긴 context 처리에 유리하다.
- 최신 LLM에서 자주 사용된다.

## 5. Tokenizer와 텍스트 데이터 구성

### 5.1 Tokenizer의 역할

Tokenizer는 입력 문장을 토큰 단위로 나누고 각 토큰에 고유 ID를 부여한다.

```python
ids = tokenizer.encode(text)
text = tokenizer.decode(ids)
```

시험에서 묻기 쉬운 차이:

- `encode`: text -> token ids
- `decode`: token ids -> text
- special token: 모델 제어, 입력/출력 구조 표시
- padding token: batch 길이를 맞추지만 일반적으로 loss 학습 대상에서는 제외

### 5.2 BPE Tokenizer

BPE(Byte Pair Encoding)는 가장 많이 등장하는 인접 token 쌍을 반복적으로 병합한다.

핵심 단계:

1. 문자 또는 작은 단위로 vocabulary를 시작한다.
2. corpus에서 인접 pair의 빈도를 센다.
3. 가장 빈도 높은 pair를 하나의 token으로 병합한다.
4. 정해진 횟수 또는 vocabulary 크기까지 반복한다.

시험 포인트:

- 자주 등장하는 단어 조각은 길게 묶여 압축 효율이 좋아진다.
- 드문 단어도 subword 조합으로 표현할 수 있다.
- GPT 계열에서는 BPE 기반 tokenizer가 자주 쓰인다.

### 5.3 WordPiece Tokenizer

WordPiece는 단순 빈도보다 “쌍으로 등장할 확률이 개별 등장 확률보다 얼마나 의미 있는가”를 본다.

- BERT 계열에서 유명하다.
- `ng`가 각각 `n`, `g`로 나올 때보다 함께 나올 확률이 높으면 병합 후보가 된다.
- 확률적 기준이 포함된 subword vocabulary 구축 방식으로 이해하면 된다.

### 5.4 Unigram Tokenizer

Unigram은 가능한 token 조합 후보 중 확률적으로 가장 좋은 조합을 선택한다.

- 처음에는 큰 vocabulary로 시작한다.
- 쓸모없는 token을 제거하면서 vocabulary를 줄인다.
- 같은 단어도 확률에 따라 다른 tokenization 후보가 가능하다.
- 통계적 특성을 잘 반영하지만 속도는 상대적으로 느릴 수 있다.

### 5.5 Special Token

Special token은 모델을 제어하고 입력/출력 구조를 정의하는 제어 신호다.

예:

- `<|endoftext|>`: 문서 또는 문장 끝
- padding token: batch 길이 맞춤
- instruction delimiter: `### Instruction:`, `### Response:` 같은 구조 표시

주의:

- special token은 모델마다 다를 수 있다.
- padding token은 보통 loss 계산에서 제외한다.

## 6. Dataset, DataLoader, Sliding Window

### 6.1 GPTDataset의 목적

GPTDataset은 원본 텍스트를 token id로 바꾼 뒤 일정 길이의 input/target 쌍으로 만든다.

```python
token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

for i in range(0, len(token_ids) - max_length, stride):
    input_chunk = token_ids[i : i + max_length]
    target_chunk = token_ids[i + 1 : i + max_length + 1]
```

시험 포인트 ★ — **Self-supervised 의 의미**:

- 사람이 별도로 정답 라벨을 달지 않아도, 텍스트 *자체가* input 과 target 의 쌍을 자동으로 만들어 준다.
- target 은 input 보다 정확히 *1 칸 뒤로 밀린* 시퀀스다. 같은 길이를 유지하면서 모든 위치마다 *다음 토큰* 을 맞히도록 학습한다.
- 즉 학습 데이터는 `(token_ids[i:i+T], token_ids[i+1:i+T+1])` 쌍의 모음이다. label 이 따로 없으니 *self-supervised* 라 부른다.
- `__getitem__` 은 같은 idx 의 input/target 쌍을 반환한다 (`return self.input_ids[idx], self.target_ids[idx]`).

### 6.2 max_length와 stride

- `max_length`: 한 번에 모델에 넣는 context 길이
- `stride`: 다음 sample을 만들 때 이동하는 간격

예:

```text
token_ids = [0, 1, 2, 3, 4, 5, 6]
max_length = 4
stride = 2

input 1 = [0, 1, 2, 3]
target1 = [1, 2, 3, 4]

input 2 = [2, 3, 4, 5]
target2 = [3, 4, 5, 6]
```

### 6.3 DataLoader

DataLoader는 Dataset에서 샘플을 가져와 batch 단위로 묶는다.

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)
```

- `shuffle=True`: 학습 순서 편향을 줄인다.
- `drop_last=True`: 마지막 batch 크기가 다를 때 버린다. shape 안정성에 유리하다.

## 7. 언어 모델 구조: RNN에서 Transformer로

### 7.1 RNN 기반 언어 모델의 한계

RNN은 이전 hidden state를 순차적으로 넘기며 처리한다.

단점:

- 이전 계산이 끝나야 다음 계산을 할 수 있어 병렬화가 어렵다.
- sequence가 길어지면 메모리와 gradient 문제가 생긴다.
- hidden state에 여러 정보가 섞여 필요한 정보를 분리하기 어렵다.

### 7.2 Attention의 도입

Attention은 입력 token들 사이의 관련도를 직접 계산해 필요한 정보를 선택한다.

- 어떤 단어를 주목할지 학습한다.
- 관련 있는 token끼리 attention score가 커진다.
- sequence 내 token 간 관계를 행렬 연산으로 병렬 처리할 수 있다.

### 7.3 Transformer

Transformer의 핵심은 Attention과 병렬화 가능한 구조다.

```text
Transformer = Attention + Feed Forward Network + Residual + LayerNorm
```

GPT는 Transformer의 Decoder만 사용한다.

## 8. Attention 상세

### 8.1 Q, K, V의 역할

입력 embedding `x`에 서로 다른 weight를 곱해 Query, Key, Value를 만든다.

```python
queries = W_query(x)
keys = W_key(x)
values = W_value(x)
```

역할:

- Query: 현재 token이 무엇을 찾는지 나타내는 질문
- Key: 각 token이 어떤 특성을 갖는지 나타내는 검색 대상
- Value: 실제로 전달될 정보

Q/K/V를 분리하는 이유:

- Q와 K를 분리하지 않으면 “질문하는 관점”과 “검색되는 기준”이 섞인다.
- K와 V를 분리하지 않으면 “찾는 기준”과 “전달할 정보”가 섞인다.
- 분리된 projection을 학습하면 같은 token도 상황에 따라 다른 역할을 할 수 있다.

시험 포인트 — 학습되는 파라미터와 shape:

- `W_query`, `W_key`, `W_value`는 모두 `nn.Linear(D_in, D_out, bias=False)` 형태로, weight shape는 `[D_in, D_out]`이다. 이 세 행렬이 attention 모듈 안에서 *학습되는* 파라미터다.
- 입력 X가 `[B, T, D_in]`이면 Q, K, V는 각각 `[B, T, D_out]`이 된다. Multi-head일 때 `D_out`을 `H`개의 head로 쪼개므로 `D_out = H · d_h`로 둔다.
- 같은 X에서 세 행렬로 따로 projection하므로, Q는 "이 위치가 알고 싶어 하는 것", K는 "각 위치가 무슨 정보를 갖고 있는지", V는 "그 위치에서 가져올 실제 값"이라는 세 역할이 분리된다.

시험 포인트 ★ — **nn.Linear forward 호출 흐름** (시험에서 자주 묻는 부분):

```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)   # __init__ 에서 한 번
queries = self.W_query(x)                              # forward 안에서 매 step 호출
```

- `self.W_query(x)` 는 PyTorch 의 `nn.Module.__call__` 을 호출하고, 그것이 다시 내부의 `forward(x)` 를 실행한다. 즉 `forward` 를 *직접* 호출하지 않는 것은, hook 과 device 이동 같은 부가 처리가 `__call__` 안에서 일어나기 때문이다.
- 내부 연산은 `x @ W.T + b`. PyTorch 저장 형태로 `W` 의 shape 가 `[d_out, d_in]` 이라 transpose 가 들어간다.
- 같은 `x` 에 대해 `W_query`, `W_key`, `W_value` 셋을 *따로* 호출하므로 Q, K, V 가 세 개의 다른 projection 결과로 나온다. 합쳐서 하나의 행렬로 두면 세 역할이 같은 부분공간에 묶여 표현력이 떨어진다.

### 8.2 Scaled Dot-Product Attention

```python
attn_scores = queries @ keys.transpose(-2, -1)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
context_vec = attn_weights @ values
```

의미:

1. Query와 Key의 dot product로 관련도 점수를 만든다.
2. `sqrt(d_k)`로 나누어 score scale을 안정화한다.
3. softmax로 확률분포를 만든다.
4. attention weight로 Value를 가중합해 context vector를 만든다.

시험 포인트 — 단계별 shape 트레이스 (single-head 기준):

```text
Q, K, V              : [B, T, D]
Q @ K^T              : [B, T, T]      # 모든 (i, j) 쌍의 점수
/sqrt(d_k) 후 softmax : [B, T, T]      # 행 단위로 합 1
weights @ V          : [B, T, D]      # context vector
```

핵심:

- 중간의 `[B, T, T]` 행렬이 attention map이다. 이 행렬의 i행 j열은 "i번째 query가 j번째 key를 얼마나 본다"를 의미한다.
- 마지막에 `@ V`까지 곱하면 token 수와 차원이 모두 원래 `[B, T, D]`로 돌아온다. 모듈 입출력 shape이 같으므로 residual `x + attn(x)`이 가능하다.

시험 포인트 ★ — **왜 Softmax 를 쓰는가?**

attention score `[B, T, T]` 를 그대로 V 에 곱하지 않고 굳이 softmax 를 통과시키는 이유는 네 가지로 정리된다.

1. **확률 분포로 만들기 위해.** softmax 후 각 query 위치(행) 의 합이 *정확히 1* 이 된다. 그래서 attention 을 "각 key 위치를 얼마나 참고할지의 확률 가중치" 로 해석할 수 있고, `attn_weights @ V` 가 *가중 평균* 이 된다.
2. **모든 가중치를 양수로 만들기 위해.** `exp(·)` 가 음수 점수를 작은 양수로, 큰 양수를 더 큰 양수로 바꾼다. 음수 가중치로 V 의 정보가 "빼지는" 일이 없다.
3. **점수 차이를 비선형적으로 강조하기 위해.** 큰 점수는 더 크게, 작은 점수는 더 작게 분포되어 *유사한 token 에 집중* 하기 좋다.
4. **mask 효과를 깔끔하게 만들기 위해.** mask 로 `-inf` 가 들어간 위치는 `exp(-inf) = 0` 이라 softmax 후 정확히 0 이 된다. 미래 token 이 attention 결과에서 *자동으로* 제거된다.

만약 softmax 를 안 쓰면? 음수·양수가 섞인 가중치가 그대로 V 에 곱해져 합이 1 이 아닌 가중합이 되고, 가중 평균의 의미가 사라져 학습이 불안정해진다.

### 8.3 왜 `sqrt(d_k)`로 나누는가

Key/Query 차원 `d_k`가 커질수록 dot product 값의 분산이 커진다. 값이 너무 커지면 softmax가 한 token에만 거의 1을 주는 형태로 뾰족해져 gradient가 불안정해질 수 있다. 그래서 `sqrt(d_k)`로 나눠 scale을 맞춘다.

시험 포인트 — 한 단계 더 자세히:

- Q, K 의 원소를 독립적 평균 0·분산 1의 분포로 본다고 가정하면, 점곱 `Q · K = Σ q_i · k_i` 는 항이 `d_k` 개라서 분산이 약 `d_k` 에 비례한다.
- 분산이 큰 값을 그대로 softmax 에 넣으면 가장 큰 logit 이 다른 logit 들을 완전히 압도해 출력 분포가 거의 one-hot 형태로 *sharp* 해진다. 이때 다른 위치들의 gradient 는 0 에 가까워져 학습이 진행되지 않는다.
- `sqrt(d_k)` 로 나눠 분산을 약 1 부근으로 되돌리면 softmax 가 적절한 스무드함을 유지해 안정적 학습이 가능하다.

### 8.4 Causal Mask

GPT는 미래 token을 보면 안 된다. 학습 시에는 전체 정답 sequence가 이미 있지만, 각 위치의 예측은 자기 앞 token만 보고 해야 한다.

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

`-torch.inf`로 채운 위치는 softmax 이후 확률이 0이 된다.

### 8.5 Attention Map

Attention map은 token 간 attention score 또는 attention weight를 행렬로 시각화한 것이다.

- 행: 현재 token 또는 query 위치
- 열: 참조되는 token 또는 key 위치
- 값이 클수록 해당 token을 많이 참고한다.

### 8.6 Multi-head Attention

Multi-head Attention은 여러 head가 서로 다른 관점에서 문맥을 파악하게 한다.

shape 흐름:

```text
x:         [B, T, D]
Q/K/V:     [B, T, D]
split:     [B, T, H, head_dim]
transpose: [B, H, T, head_dim]
scores:    [B, H, T, T]
context:   [B, H, T, head_dim]
combine:   [B, T, D]
```

코드 핵심:

```python
keys = keys.view(b, num_tokens, num_heads, head_dim).transpose(1, 2)
queries = queries.view(b, num_tokens, num_heads, head_dim).transpose(1, 2)
values = values.view(b, num_tokens, num_heads, head_dim).transpose(1, 2)

attn_scores = queries @ keys.transpose(2, 3)
context_vec = (attn_weights @ values).transpose(1, 2)
context_vec = context_vec.contiguous().view(b, num_tokens, d_out)
```

시험 포인트 — head 분리·합치기 순서 한눈에:

```text
[B, T, D]
  ──view(B, T, H, d_h)──>      [B, T, H, d_h]
  ──transpose(1, 2)──>         [B, H, T, d_h]
  ── Q@K^T / softmax / @V ──>  [B, H, T, d_h]
  ──transpose(1, 2)──>         [B, T, H, d_h]
  ──contiguous().view──>       [B, T, D]
```

- `D = H · d_h` 이므로 head 분리는 차원을 *나누는* 일이지 *늘리는* 일이 아니다. 파라미터 수는 single-head 와 거의 같다.
- `contiguous()` 는 `transpose` 가 만든 비연속 메모리 상태를 풀어 view 가 가능하게 한다.

### 8.7 Attention vs FFN 역할 비교

GPT 한 블록 안의 두 sublayer는 역할이 명확히 갈린다.

| 구분 | Attention | FFN |
|---|---|---|
| 정보 흐름 | token 사이 (mixing across positions) | token 내부 (per-position 비선형 변환) |
| 입력 의존성 | 다른 token 의 K, V 와 상호작용 | 자기 위치의 hidden 만 처리 |
| 활성함수 | softmax (가중치) | GELU (비선형성) |
| 파라미터 | `W_Q, W_K, W_V, W_O` | `W_1 (D→4D)`, `W_2 (4D→D)` |
| 차원 변화 | 입출력 모두 `D` | 내부에서 `4D` 로 확장 후 `D` 로 복원 |

직관:

- Attention 은 "지금 위치가 다른 위치의 무엇을 끌어올지" 를 결정하고,
- FFN 은 끌어온 정보를 위치별로 비선형적으로 가공해 다음 블록으로 넘긴다.

## 9. GPT 모델 구성 요소

### 9.1 GPT 전체 구조

```text
Token IDs
-> Token Embedding
-> Positional Embedding 추가
-> Dropout
-> TransformerBlock 반복
-> Final LayerNorm
-> Linear Output Head
-> Logits
```

시험 포인트 — End-to-End shape 트레이스:

```text
in_idx           : [B, T]
tok_emb(in_idx)  : [B, T, D]
pos_emb(arange)  : [T, D]            ─┐
tok + pos        : [B, T, D]          │ broadcasting
drop_emb         : [B, T, D]          │
trf_blocks × N   : [B, T, D]          │ 모듈 입출력 동일
final_norm       : [B, T, D]          │
out_head (Linear): [B, T, V]          │ V = vocab_size
logits[:, -1, :] : [B, V]             │ 다음 token 후보 점수
argmax(dim=-1)   : [B, 1]             ─┘ 다음 token ID
```

핵심:

- `D` 가 입력 임베딩부터 마지막 LayerNorm 까지 끝까지 유지된다. 그래야 residual 합산이 가능하고 같은 블록을 N 번 쌓을 수 있다.
- 차원이 바뀌는 곳은 *시작* (`vocab_size → D`) 과 *끝* (`D → vocab_size`) 두 군데뿐이다.

### 9.2 TransformerBlock

강의와 실습의 GPT block은 Pre-LayerNorm 구조로 이해하면 된다.

```text
input x
-> LayerNorm
-> MultiHeadAttention
-> Dropout
-> Residual Add
-> LayerNorm
-> FeedForward
-> Dropout
-> Residual Add
```

코드 흐름:

```python
shortcut = x
x = self.norm1(x)
x = self.att(x)
x = self.drop_shortcut(x)
x = x + shortcut

shortcut = x
x = self.norm2(x)
x = self.ff(x)
x = self.drop_shortcut(x)
x = x + shortcut
```

시험 포인트 — pre-LN vs post-LN:

- **pre-LN** (위 코드): LayerNorm 이 sublayer (`att`, `ff`) *바로 앞* 에 와서 sublayer 입력이 항상 정규화된 상태로 들어간다. 깊은 모델에서도 gradient 가 발산하지 않아 *학습 안정성* 이 좋다. GPT-2 이후 표준.
- **post-LN** (원래 Transformer 논문): sublayer 뒤에 LayerNorm 이 오는 구조. 학습률을 작게 잡거나 warmup 을 길게 해야 안정적이다.
- Dropout 은 residual *합산 직전* 에 적용한다(`drop_shortcut(att(x))` 뒤). 이로써 같은 블록을 여러 번 통과해도 활성도가 폭주하지 않는다.

### 9.3 Layer Normalization

LayerNorm은 각 sample의 feature 차원에서 평균과 분산을 구한다.

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (x - mean) / torch.sqrt(var + eps)
out = scale * norm_x + shift
```

시험 포인트:

- LLM에서는 BatchNorm보다 LayerNorm을 주로 사용한다.
- gamma/scale과 beta/shift는 학습 가능한 파라미터다.
- 정규화가 항상 최선은 아니므로 모델이 필요한 범위로 되돌릴 수 있게 scale/shift를 학습한다.

### 9.4 Residual Connection

Residual connection은 입력을 출력에 더한다.

```python
x = block(x) + shortcut
```

효과:

- 깊은 모델에서 gradient 흐름이 안정적이다.
- 모든 것을 새로 학습하기보다 변화량을 학습하게 한다.
- Transformer block을 여러 층 쌓기 쉽게 한다.

### 9.5 Feed Forward Network와 GELU

GPT 계열 FFN은 보통 embedding 차원을 4배로 확장한 뒤 다시 줄인다.

```python
nn.Sequential(
    nn.Linear(emb_dim, 4 * emb_dim),
    GELU(),
    nn.Linear(4 * emb_dim, emb_dim),
)
```

역할:

- Attention이 token 간 정보를 섞는다면 FFN은 각 token 위치에서 feature를 비선형 변환한다.
- GELU는 GPT 계열에서 많이 쓰이는 activation이다.

#### 9.5b Sigmoid / ReLU / GELU 비교

강의 PDF 에 등장하는 세 activation 의 차이.

| 함수 | 출력 범위 | 특징 | 대표 사용처 |
|---|---|---|---|
| Sigmoid | `[0, 1]` | 이진 분류·gate. 큰 입력에서 gradient 가 0 으로 죽음 (saturation). | BCE 출력, gating |
| ReLU | `[0, ∞)` | 양수만 통과, 음수는 0. 빠르지만 *dead neuron* 위험. | 일반 NN, CNN |
| GELU | 약 `(−0.17, ∞)` | Sigmoid 와 비슷한 곡선으로 *smooth* 하게 음수도 일부 통과. | GPT 계열 FFN |

직관:

- ReLU 는 0 미만을 완전히 잘라내 정보 손실이 있을 수 있다.
- GELU 는 `x · Φ(x)` (Φ 는 표준 정규분포 누적함수) 로, 음수에서도 점진적으로 작은 값을 통과시킨다. 학습 안정성과 성능에서 GPT 류에 잘 맞는다.

### 9.6 Output Head와 logits

Output head는 hidden vector를 vocabulary 크기의 logits로 바꾼다.

```python
self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)
logits = self.out_head(x)
```

shape:

```text
x:      [B, T, D]
logits: [B, T, V]
```

### 9.7 왜 Transformer module 입출력 차원이 같은가

강의 PDF의 핵심 포인트:

- module을 계속 쌓기 위해 입출력 차원이 같아야 한다.
- residual connection을 하려면 더하는 두 tensor shape가 같아야 한다.

추가 설명 (시험에서 자주 묻는 두 가지를 함께 외우기):

1. **Stacking 가능성**: `BlockN(Block_{N-1}(...))` 처럼 같은 모양의 블록을 N 번 합성하려면 `f: ℝ^D → ℝ^D` 형태여야 한다.
2. **Residual 합산 가능성**: `x + sublayer(x)` 의 두 항 shape 가 같아야 element-wise 덧셈이 성립한다. 차원이 다르면 별도 projection 이 필요해 모델이 복잡해진다.

즉 "모듈 입출력 차원을 `D` 로 고정"하는 것은 *깊이 있게 쌓을 수 있도록* 하기 위한 설계 결정이다.

## 10. 텍스트 생성과 Temperature

### 10.1 Greedy Decoding

텍스트 생성 시에는 마지막 위치의 logits만 보고 다음 token을 고른다.

```python
idx_cond = idx[:, -context_size:]
with torch.no_grad():
    logits = model(idx_cond)

logits = logits[:, -1, :]
idx_next = torch.argmax(logits, dim=-1, keepdim=True)
idx = torch.cat((idx, idx_next), dim=1)
```

이 과정을 autoregressive하게 반복한다.

### 10.2 Training과 Inference 차이

Training:

- 전체 sequence를 한 번에 넣는다.
- causal mask로 미래 token을 가린다.
- 각 위치에서 next token prediction loss를 계산한다.

Inference:

- 생성된 token을 다시 입력에 붙인다.
- 한 token씩 autoregressive하게 생성한다.
- causal mask의 논리는 동일하지만 실제로는 이전 token만 context로 들어간다.

### 10.3 Temperature

> ⚠ **시험 출제 포인트 외 — 참고용 (§10.3 ~ §10.5)**
> 본 항목(Temperature, Top-K, 비교표)은 강의 자료(커리큘럼 PDF)에는 있으나 26.5월 시험 출제 포인트 PDF 키워드(LLM 모델 원리 / 데이터 구성 / RAG / On-device 3종)에 포함되지 않는다. 시험 대비는 ◎ 영역을 우선하라. (Greedy 만 ◯)

Temperature는 softmax 분포의 날카로움을 조절한다.

```text
softmax(logits / T)
```

- T가 낮으면 높은 확률 token이 더 확정적으로 선택된다.
- T가 높으면 낮은 확률 token에도 더 많은 비중이 간다.
- 창의적 생성은 temperature를 높이고, 안정적 답변은 낮추는 경향이 있다.

### 10.4 Top-K Sampling

Top-K 는 확률 상위 `K` 개 token 만 후보로 남기고 나머지를 0 으로 만든 뒤 sampling 한다.

```python
top_logits, _ = torch.topk(logits, top_k)
min_val = top_logits[:, -1]
logits = torch.where(
    logits < min_val.unsqueeze(-1),
    torch.tensor(float("-inf")).to(logits.device),
    logits,
)
probs = torch.softmax(logits / temperature, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

요약:

- `K` 가 작을수록 결정적 (Greedy 에 가까워짐), 클수록 다양성 증가.
- Temperature 와 함께 자주 묶어 쓴다. 예: `T = 0.7`, `K = 50` 이 대화형 LLM 의 흔한 기본값.
- Top-K 는 *고정된 개수* 의 후보를 본다. 분포가 평탄할 때 안전망 역할.

### 10.5 Greedy / Top-K / Temperature 한눈에

| 전략 | 결정성 | 다양성 | 비고 |
|---|---|---|---|
| Greedy (argmax) | 매우 높음 | 매우 낮음 | `T → 0` 의 극한 |
| Temperature only | T 로 조절 | T 로 조절 | 분포 평탄/뾰족 조절 |
| Top-K + Temperature | 후보 K 개로 한정 + T 로 분포 | K, T 로 동시 조절 | 실무 표준 |

## 11. 학습과 평가

### 11.1 Pretraining

GPT pretraining은 self-supervised learning이다.

- 사람이 label을 별도로 달지 않아도 텍스트 자체에서 target을 만든다.
- 목표는 다음 token을 맞히는 Causal Language Modeling이다.
- loss는 vocabulary 전체에 대한 Cross Entropy다.

### 11.2 Dataset 분리

일반적인 분리:

- train: 모델 파라미터 학습
- validation: overfitting 확인, hyperparameter 조정
- test: 최종 일반화 성능 평가

### 11.3 Classification Fine-tuning

분류 fine-tuning에서는 GPT의 output head를 class 수에 맞게 바꾼다.

```python
model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)
```

분류 흐름:

```text
text -> token ids -> GPT -> logits -> softmax -> argmax -> class
```

평가 metric:

- accuracy: 전체 중 맞춘 비율
- precision: positive라고 예측한 것 중 실제 positive 비율
- recall: 실제 positive 중 찾아낸 비율
- F1: precision과 recall의 조화평균
- loss: 모델 예측 확률과 정답의 차이

### 11.4 Loss 함수와 Optimizer 비교

강의 PDF 에서 명시적으로 다루는 항목들을 한 곳에 모은 정리.

**Loss 함수**

| Loss | 출력 가정 | 정답 형태 | 사용처 |
|---|---|---|---|
| MSE | 연속 실수 | 실수 | 회귀 (regression) |
| BCE (Binary Cross Entropy) | sigmoid 후 확률 (0~1) | 0 또는 1 | 이진 분류, 다중 라벨 |
| Cross Entropy (multi-class) | softmax 후 확률 분포 | class index | 다중 클래스 분류, LM next-token |

핵심:

- `nn.CrossEntropyLoss` 는 내부적으로 `log_softmax + nll_loss` 를 한 번에 한다. 즉 모델의 *logits* 을 그대로 넣어야 한다 (softmax 를 직접 적용하지 말 것).
- `ignore_index=-100` 을 지정하거나 target 위치를 `-100` 으로 두면 해당 위치는 loss 계산에서 제외된다 (instruction fine-tuning 의 prompt mask 가 정확히 이 메커니즘이다).

**Optimizer**

> ⚠ **시험 출제 포인트 외 — 참고용**
> Optimizer (SGD/Momentum/RMSprop/Adam) 는 26.5월 시험 출제 포인트 PDF 키워드에 등장하지 않는다. Loss 함수 표는 ◎ 라 그대로 두지만, 아래 Optimizer 표·학습 단위는 참고용이다.

| Optimizer | 핵심 아이디어 |
|---|---|
| SGD | 한 batch 의 gradient 로 직접 weight 갱신 |
| Momentum | 과거 gradient 의 *지수 이동 평균* 을 더해 진동을 줄이고 가속 |
| RMSprop | gradient 의 *제곱 평균* 으로 좌표별 학습률 조정 |
| Adam | Momentum + RMSprop. *1차 모멘트* (평균) 와 *2차 모멘트* (제곱 평균) 를 모두 추정 |

학습 단위:

- `epoch` : 학습 데이터 전체를 한 번 통과
- `step` (iteration) : 한 batch 에 대해 weight 한 번 갱신
- `batch_size` : 한 step 에 사용하는 sample 수
- learning rate scheduling : `step decay` (epoch 마다 일정 비율 감소), `cosine annealing` 등이 표준

PyTorch 학습 루프 한 줄 요약:

```python
optimizer.zero_grad()    # 이전 step gradient 초기화
loss = compute_loss(...)
loss.backward()          # gradient 계산
optimizer.step()         # weight 갱신
```

## 12. Fine-tuning, LoRA, QLoRA

> ⚠ **시험 출제 포인트 외 — 참고용 (§12 전체)**
> LoRA / QLoRA / PEFT 는 26.5월 시험 출제 포인트 PDF 키워드에 등장하지 않는다. 강의 자료(커리큘럼)에는 있으나 시험 대비는 ◎ 영역을 우선하라.

### 12.1 Full Fine-tuning과 PEFT

Full fine-tuning:

- 모델 전체 파라미터를 업데이트한다.
- 성능은 좋을 수 있지만 비용과 메모리 요구가 크다.

PEFT(Parameter-Efficient Fine-Tuning):

- 전체 모델 대신 일부 작은 파라미터만 학습한다.
- LoRA가 대표적이다.

### 12.2 LoRA의 기본 아이디어

LoRA는 기존 weight `W`를 직접 바꾸지 않고 low-rank update `Delta W = A @ B`를 더한다.

```python
output = linear(x) + (alpha / rank) * (x @ A @ B)
```

행렬 shape:

```text
A: [in_dim, rank]
B: [rank, out_dim]
x @ A @ B: [batch..., out_dim]
```

### 12.3 LoRA 초기화

강의 PDF의 중요한 포인트:

- A는 랜덤 초기화한다.
- B는 0으로 초기화한다.
- A와 B를 모두 random으로 초기화하면 처음부터 기존 모델 출력이 바뀐다.
- A와 B를 모두 0으로 초기화하면 gradient가 제대로 흐르지 않아 업데이트가 어렵다.
- 첫 번째 역전파에서는 B가 먼저 업데이트되고, 이후 A도 학습된다.

### 12.4 LoRA 학습 절차

```python
for param in model.parameters():
    param.requires_grad = False

replace_linear_with_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
```

- 기존 파라미터는 동결한다.
- Linear layer를 LoRA가 포함된 wrapper로 교체한다.
- optimizer에는 전체 model parameters를 넘겨도 `requires_grad=True`인 LoRA 파라미터만 업데이트된다.

### 12.5 QLoRA

QLoRA는 LLM weight를 4-bit 등 낮은 bit로 양자화해 메모리 부담을 줄이고, LoRA adapter를 학습한다.

핵심:

- 모델 weight 저장 메모리를 줄인다.
- 학습 계산에는 dequantization 과정이 포함된다.
- 결과 계산은 BF16 같은 형식과 연결될 수 있다.
- 큰 모델을 한정된 GPU 메모리에서 fine-tuning 가능하게 한다.

## 13. Instruction Fine-tuning

### 13.1 Instruction 데이터 포맷

실습에서는 Alpaca 스타일 포맷을 사용한다.

```text
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
...

### Input:
...

### Response:
...
```

- `Instruction`: 해야 할 작업
- `Input`: 추가 입력 또는 context, 없을 수 있음
- `Response`: 모델이 생성해야 할 정답 응답

### 13.2 InstructionDataset

각 entry를 하나의 full text로 만든 뒤 tokenization한다.

```python
instruction_plus_input = format_input(entry)
response_text = f"\n\n### Response:\n{entry['output']}"
full_text = instruction_plus_input + response_text
encoded = tokenizer.encode(full_text)
```

### 13.3 Collate와 padding mask

가변 길이 sequence를 batch로 묶으려면 padding이 필요하다.

```python
new_item += [pad_token_id]
padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
inputs = torch.tensor(padded[:-1])
targets = torch.tensor(padded[1:])
```

padding token을 loss에서 제외:

```python
mask = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index
```

`ignore_index=-100`은 PyTorch `CrossEntropyLoss`가 해당 위치를 loss 계산에서 무시하게 한다.

## 14. DPO: Direct Preference Optimization

> ⚠ **시험 출제 포인트 외 — 참고용 (§14 전체)**
> DPO / RLHF / PPO 는 26.5월 시험 출제 포인트 PDF 키워드에 등장하지 않는다. 강의 자료(커리큘럼)에는 있으나 시험 대비는 ◎ 영역을 우선하라.

### 14.1 RLHF와 DPO

RLHF 또는 DPO는 인간이 무엇을 더 선호하는지 모델에 가르치는 방식이다.

예:

```text
Prompt: 여름 휴가지를 추천해줘
Chosen: 제주도 함덕 해변은 어떠신가요? ...
Rejected: 바다로 가세요.
```

DPO는 복잡한 강화학습 과정이나 별도 reward model 없이 선호 데이터만으로 직접 학습한다.

#### 14.1b RLHF (Reward Model + PPO) 가 무엇이고 DPO 가 어떻게 단순화하는가

전통적 RLHF 는 두 단계로 나뉜다.

1. **Reward Model 학습**: 사람이 평가한 `(Prompt, Chosen, Rejected)` 데이터로 *별도 보상 모델* `r_φ(x, y)` 을 학습한다. 이 모델은 (prompt, response) 한 쌍을 입력받아 점수 하나를 출력한다.
2. **PPO 로 정책 최적화**: 본체 LLM (policy) 이 reward model 의 점수를 최대화하도록 강화학습 (Proximal Policy Optimization) 으로 학습한다. 동시에 reference model 과의 KL 거리가 너무 멀어지지 않도록 KL constraint 를 둔다.

문제점:

- Reward model 학습이 별도로 필요하고 노이즈가 그대로 정책에 전파된다.
- PPO 는 hyper-parameter 에 민감하고 학습이 불안정하기 쉽다.
- 메모리도 많이 든다 (policy + reference + reward model 동시 보관).

**DPO 의 단순화**:

- Bradley-Terry 모델로 선호 확률을 표현해 reward 함수를 *암시적으로* 만들고, 최적 정책의 closed-form 을 유도한다.
- 그 결과 reward model 도 PPO 도 없이, *지도학습 형태의 loss* 한 줄로 정책을 직접 학습할 수 있다.
- Loss: `−logsigmoid( β · ( log π_θ(y_w|x) − log π_θ(y_l|x) − log π_ref(y_w|x) + log π_ref(y_l|x) ) )` 형태. β 가 KL 강도를 조절한다.

요약 한 줄: *"DPO 는 RLHF 의 reward 학습 + PPO 단계를 하나의 cross-entropy 류 loss 로 압축한 것"*.

### 14.2 DPO 데이터 구조

Reward model 데이터:

```text
Prompt, Response, Feedback 또는 Rating
```

DPO 데이터:

```text
Prompt, Chosen, Rejected
```

### 14.3 Policy Model과 Reference Model

- `policy_model`: 학습되어 업데이트되는 모델
- `reference_model`: 고정된 기준 모델, 보통 SFT가 끝난 모델

DPO는 policy가 reference보다 chosen을 rejected보다 더 선호하도록 학습한다.

### 14.4 DPO에서 prompt mask가 필요한 이유

DPO의 관심은 “질문을 얼마나 잘 생성하느냐”가 아니라 “같은 prompt에 대해 어떤 response를 더 선호하느냐”이다. 그래서 prompt 부분은 loss 계산에서 제외하고 response 부분만 log probability 비교에 사용한다.

```python
if mask_prompt_tokens:
    mask[:prompt.shape[0] + 2] = False
```

### 14.5 Log probability 계산

Autoregressive 모델은 입력의 각 위치에서 다음 token을 예측하므로 labels와 logits를 shift한다.

```python
labels = labels[:, 1:].clone()
logits = logits[:, :-1, :]
log_probs = F.log_softmax(logits, dim=-1)
selected_log_probs = torch.gather(
    input=log_probs,
    dim=-1,
    index=labels.unsqueeze(-1)
).squeeze(-1)
```

mask가 있으면 유효한 response token만 평균낸다.

### 14.6 DPO Loss

```python
model_logratios = model_chosen_logprobs - model_rejected_logprobs
reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
logits = model_logratios - reference_logratios
losses = -F.logsigmoid(beta * logits)
```

해석:

- policy가 chosen을 rejected보다 더 높게 평가하면 `model_logratios`가 커진다.
- reference 대비 policy가 선호 차이를 더 잘 만들면 loss가 작아진다.
- `beta`는 reference에서 얼마나 벗어날지 조절하는 하이퍼파라미터다.

## 15. RAG

### 15.1 RAG의 기본 개념

RAG(Retrieval-Augmented Generation)는 외부 문서를 검색해 LLM 입력 context로 넣고, 모델이 그 근거를 바탕으로 답하게 하는 구조다.

기본 흐름:

```text
문서 수집
-> chunking
-> embedding
-> vector index 저장
-> query embedding
-> similarity search / retrieval
-> retrieved context + prompt 구성
-> LLM generation
```

### 15.2 주요 구성 요소

- Document loader: PDF, HTML, DB 등에서 문서를 불러온다.
- Chunker: 긴 문서를 작은 조각으로 나눈다.
- Embedding model: chunk와 query를 벡터로 변환한다.
- Vector store/index: embedding을 저장하고 검색한다.
- Retriever: query와 유사한 chunk를 찾아온다.
- Prompt template: retrieved context와 user query를 LLM 입력 형식으로 구성한다.
- Generator: LLM이 최종 답변을 생성한다.

### 15.3 RAG와 Fine-tuning 차이

RAG:

- 모델 weight를 바꾸지 않는다.
- 외부 지식을 검색해 prompt에 넣는다.
- 최신 사내 문서, 규정, FAQ 등 자주 바뀌는 지식에 유리하다.

Fine-tuning:

- 모델 파라미터를 업데이트한다.
- 특정 형식, 스타일, 태도, task 패턴을 학습시키는 데 유리하다.
- 최신 지식을 계속 반영하려면 재학습이 필요할 수 있다.

### 15.4 Chunk size와 overlap

Chunk size가 너무 작으면:

- 필요한 문맥이 잘려 답변 근거가 부족해질 수 있다.

Chunk size가 너무 크면:

- 검색 결과가 덜 정확해질 수 있다.
- context window를 낭비할 수 있다.
- irrelevant text가 많이 섞일 수 있다.

Overlap:

- chunk 경계에서 문맥이 끊기는 문제를 줄인다.
- 너무 크면 중복 저장과 검색 비용이 증가한다.

### 15.5 llama-index와 MCP

llama-index:

- 문서 loading, indexing, retriever, query engine 구성에 쓰이는 RAG 프레임워크로 이해한다.
- 문서를 LLM이 검색하기 쉬운 구조로 바꾸는 데 초점이 있다.

MCP:

- 모델 또는 애플리케이션이 외부 도구, 데이터 소스와 표준화된 방식으로 연결되도록 하는 프로토콜 관점으로 이해한다.
- RAG 도구, 사내 시스템, 파일, DB 등을 모델이 사용할 수 있게 연결하는 맥락에서 출제될 수 있다.

## 16. 최신 Attention/최적화 기법

> ⚠ **시험 출제 포인트 외 — 참고용 (§16 전체)**
> Sliding Window Attention / Gated DeltaNet / Flash Attention / KV cache / GQA / MLA / MoE 는 모두 26.5월 시험 출제 포인트 PDF 키워드에 등장하지 않는다. 강의 자료(커리큘럼)에는 있으나 시험 대비는 ◎ 영역(LLM 모델 원리·데이터 구성, RAG, On-device 3종)을 우선하라.

### 16.1 Sliding Window Attention

Sliding window attention은 전체 context를 모두 보지 않고 일정 길이의 앞 token만 본다.

- attention 비용을 줄인다.
- 매우 긴 sequence에서 메모리 사용량을 줄일 수 있다.
- 단점은 window 밖의 먼 정보는 직접 참고하기 어렵다는 점이다.

### 16.2 Gated DeltaNet

강의 PDF에서는 Qwen3-Next와 Kimi Linear에서 사용하는 attention 메커니즘으로 소개된다.

핵심:

- 어떤 정보가 중요하며 얼마나 통과시킬지 동적으로 결정한다.
- linear attention 계열의 효율성과 gated attention의 선택성을 결합하는 hybrid 구조로 이해한다.
- Update Gate, Decay Gate, Output Gate가 등장한다.

Gate 의미:

- Decay Gate: 과거 정보를 얼마나 유지하거나 잊을지 결정
- Update Gate: 현재 새 정보를 메모리에 얼마나 반영할지 결정
- Output Gate: 어떤 정보를 얼마나 출력으로 통과시킬지 결정
- Q와 K를 L2 정규화해 수치 안정성을 확보한다.

### 16.3 Flash Attention

Flash Attention은 attention 수식을 근사하는 것이 아니라 GPU memory I/O를 줄이는 최적화다.

핵심:

- attention 계산은 GPU의 빠른 SRAM에서 수행한다.
- 큰 중간 행렬을 HBM에 자주 읽고 쓰는 일을 줄인다.
- tiling을 통해 필요한 block만 SRAM으로 가져온다.
- online softmax로 softmax 계산에 필요한 값을 한 loop 안에서 통합한다.
- 학습 속도를 높이고 메모리 사용량을 줄인다.

시험 포인트:

- Flash Attention은 approximate attention이 아니다.
- 핵심은 I/O 최적화다.
- tiling, SRAM, HBM, online softmax를 연결해서 설명할 수 있어야 한다.

### 16.4 KV Cache

Inference에서는 이전 token들의 Key/Value를 계속 재사용할 수 있다. KV cache는 이미 계산한 K/V를 저장해 다음 token 생성 시 반복 계산을 줄인다.

- 장점: autoregressive inference 속도 향상
- 단점: context가 길어질수록 cache 메모리 증가
- quantization과 결합해 KV cache 메모리를 줄일 수 있다.

### 16.5 GQA (Grouped Query Attention)

여러 Query head 가 *동일한* Key/Value head 를 공유하는 구조다. 표준 Multi-head Attention 에서는 Q, K, V head 수가 모두 동일하다(=`H`)지만 GQA 는 K, V head 수만 `G` 개로 줄이고 (예: `H = 32`, `G = 4`), 각 K/V head 를 여러 Q head 가 함께 사용한다.

- 효과: KV cache 메모리·대역폭이 `H / G` 배 절감된다. 추론 속도 ↑.
- Llama 2/3, Gemma 등 최근 모델의 표준.
- 극단인 `G = 1` 은 Multi-Query Attention (MQA) 이다.

### 16.6 MLA (Multi-head Latent Attention)

DeepSeek 계열에서 도입한 방식으로, K 와 V 를 *더 작은 latent 벡터* 로 압축해 저장한다.

- KV cache 에 원본 K, V 대신 압축된 latent 벡터만 저장 → 메모리 더 큰 절감.
- 추론 시 latent 에서 K, V 를 다시 풀어 attention 을 계산한다.
- GQA 가 head 수를 줄이는 방식이라면 MLA 는 *차원 자체* 를 줄이는 방식이다.

### 16.7 MoE (Mixture of Experts)

FFN 자리를 여러 개의 "expert" FFN 으로 바꾸고, *router* 가 token 마다 일부 expert (보통 top-2) 만 골라 활성화하는 구조.

```text
token h
  -> router -> top-K experts 선택
  -> 선택된 expert FFN 들의 가중합
  -> out
```

핵심:

- 전체 파라미터는 매우 크지만 한 token 의 *연산량* 은 활성화된 expert 만큼만 든다 (sparse activation).
- 같은 연산 비용에서 모델 capacity 를 크게 키울 수 있다.
- Mixtral, DeepSeek-V3, Qwen MoE 등이 대표적.

세 기법 한 줄 요약:

- **GQA**: K/V 의 *head 수* 줄이기 → KV cache 절감
- **MLA**: K/V 의 *차원* 압축 → KV cache 더 절감
- **MoE**: FFN 을 *여러 expert + router* 로 → 활성 연산량 절감, capacity 확장

## 17. On-device 최적화: Quantization, Pruning, Distillation

### 17.1 Quantization

Quantization은 연속적인 실수 weight 또는 activation을 낮은 bit의 이산값으로 표현한다.

목적:

- 모델 크기 감소
- 메모리 사용량 감소
- memory I/O 감소
- 추론 속도 향상

Layer 단위 관점:

```text
FP32/BF16 weight
-> scale/zero-point 등을 이용해 INT8/INT4로 양자화
-> 연산 또는 사용 시 dequantization 또는 quantized kernel 사용
```

주의:

- 메모리가 줄었다고 항상 계산이 자동으로 빨라지는 것은 아니다.
- kernel 지원, dequantization 비용, hardware 특성이 중요하다.

시험 포인트 — **layer 수준 quant/dequant 수식** (출제 포인트 직격):

```text
Quantize   :  q = round(x / s) + z         # s = scale, z = zero-point
Range clip :  q = clip(q, q_min, q_max)    # 표현 가능한 정수 범위로
Dequantize :  x' ≈ s · (q − z)             # 원래 실수 근사 복원
```

- `s` 는 양자화 단위 (예: `s = (max − min) / (q_max − q_min)`).
- `z` 는 0 이 q_min, q_max 안의 어느 정수에 대응될지 정하는 offset (`z = round(−min / s)`).
- 추론 시 weight 를 INT8/INT4 로 저장했다가 *layer 단위로* dequantize 해서 BF16/FP32 연산에 사용한다 (weight-only 양자화). activation 까지 양자화하면 INT 만으로 곱셈하는 quantized kernel 을 쓸 수도 있다.

대표 precision 비교:

| 형식 | bit | 특징 |
|---|---|---|
| FP32 | 32 | 표준 학습 정밀도 |
| BF16 | 16 | exponent 는 FP32 와 동일, mantissa 만 짧음 → 학습 안정 |
| FP16 | 16 | mantissa 큼, exponent 좁아 overflow 위험 |
| INT8 | 8 | 대표 추론 양자화. weight/activation 둘 다 가능 |
| INT4 | 4 | QLoRA 등에서 사용. 메모리 ~1/8 |

요약: *"quant 는 `s, z` 로 실수를 정수에 매핑하고, dequant 는 같은 `s, z` 로 다시 실수에 근사한다"*. 이 한 줄이 시험 직격 정답이다.

### 17.2 Pruning

Pruning은 중요도가 낮은 weight, neuron, channel, attention head 등을 제거한다.

중요도 기준 예:

- weight magnitude가 작은 값
- activation 영향이 작은 구조
- gradient 또는 loss 변화가 작은 구조

효과:

- 연산량 감소
- 모델 크기 감소
- 추론 속도 개선 가능

주의:

- 구조적 pruning이 아니면 실제 hardware 속도 개선으로 이어지지 않을 수 있다.
- pruning 후 fine-tuning으로 성능 회복이 필요할 수 있다.

시험 포인트 — **중요도 기준 3종** (출제 포인트 직격):

| 기준 | 중요도 정의 | 의미 |
|---|---|---|
| Magnitude | `|w|` (절대값) | 절대값이 작은 weight 는 출력 기여가 적다고 보고 우선 제거 |
| Gradient | `|w · ∂L/∂w|` (Taylor 1차 근사) | weight 를 0 으로 만들 때 loss 변화가 작은 것을 우선 제거 |
| Head importance | head 출력의 노름 또는 attention 기여도 | 통째로 제거 가능 → *structured pruning* |

분류:

- **Unstructured pruning** : 개별 weight 단위로 0 으로 만든다. sparsity 가 불규칙해 일반 GPU 에서 가속이 어렵다.
- **Structured pruning** : channel / head / filter 단위로 제거. 행렬 차원 자체가 줄어 HW 가속이 쉽다.
- **Sparsity ratio** : 전체 weight 중 제거한 비율. 예: 50% pruning = sparsity 0.5.

전체 흐름:

1. 중요도 기준에 따라 후보 선정
2. 해당 weight/head 제거 (또는 mask)
3. 성능 하락 보정을 위해 짧게 fine-tune

### 17.3 Distillation

Distillation은 큰 teacher 모델의 지식을 작은 student 모델로 전달한다.

학습 신호:

- hard label: 정답 label
- soft label: teacher가 출력한 확률분포
- logits: softmax 전 teacher 출력

장점:

- soft label은 class 간 상대적 유사도 정보를 담는다.
- 작은 모델이 teacher의 일반화 패턴을 배울 수 있다.
- on-device 배포에 유리하다.

일반 loss:

```text
student_loss = alpha * CE(student, hard_label)
             + (1 - alpha) * KL(student_distribution, teacher_distribution)
```

시험 포인트 — **teacher 예측을 student 학습에 "연결" 하는 방법** (출제 포인트 직격):

1. teacher 의 logits `z_t` 를 Temperature `T` 로 부드럽게 만든다.
   - `p_t = softmax(z_t / T)` → soft label
2. student 도 같은 `T` 로 부드럽게 만든다.
   - `p_s = softmax(z_s / T)`
3. 두 분포의 차이를 줄이도록 KL divergence loss 를 둔다.
   - `KL_soft = KL( p_t || p_s )`
4. 동시에 hard label 에 대한 CE loss 도 함께 둔다.
   - `CE_hard = CE(softmax(z_s), y_hard)`
5. 두 loss 를 합친다.
   - `loss = α · CE_hard + (1 − α) · T² · KL_soft`
   - `T²` 곱은 Temperature 가 gradient 크기를 작게 만드는 것을 보정한다 (Hinton 의 원 논문).

직관 — *왜 soft label 이 hard label 보다 풍부한가?*

- Hard label `[0, 0, 1, 0, 0]` 은 "정답이 무엇인가" 만 가르친다.
- Soft label `[0.05, 0.10, 0.65, 0.18, 0.02]` 은 "정답이 무엇이면서, 다른 클래스 중 어느 것이 더 비슷하고 어느 것이 무관한가" 까지 가르친다. 이 *상대적 거리* 정보가 student 의 일반화에 큰 도움이 된다.

요약: *"teacher 의 soft label 분포를 KL 로 student 에게 옮겨 붓고, hard label CE 와 가중합한다"*.

## 18. 코드 빈칸 출제 포인트

### 18.1 Dataset

```python
input_chunk = token_ids[i : i + max_length]
target_chunk = token_ids[i + 1 : i + max_length + 1]
```

### 18.2 Attention

```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

keys = self.W_key(x)
queries = self.W_query(x)
values = self.W_value(x)

attn_scores = queries @ keys.transpose(1, 2)
attn_scores.masked_fill_(mask_bool, -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
context_vec = attn_weights @ values
```

### 18.3 Multi-head

```python
x = x.view(b, num_tokens, num_heads, head_dim)
x = x.transpose(1, 2)
```

### 18.4 LayerNorm

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (x - mean) / torch.sqrt(var + self.eps)
return self.scale * norm_x + self.shift
```

### 18.5 GPTModel

```python
self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

tok_embeds = self.tok_emb(in_idx)
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
x = tok_embeds + pos_embeds
logits = self.out_head(x)
```

### 18.6 Generation

```python
with torch.no_grad():
    logits = model(idx_cond)

logits = logits[:, -1, :]
idx_next = torch.argmax(logits, dim=-1, keepdim=True)
idx = torch.cat((idx, idx_next), dim=1)
```

### 18.7 LoRA

> ⚠ **시험 출제 포인트 외 — 참고용** (§18.7 / 아래 LoRA 코드 빈칸)

```python
self.A = nn.Parameter(torch.empty(in_dim, rank))
self.B = nn.Parameter(torch.zeros(rank, out_dim))

delta = (self.alpha / self.rank) * (x @ self.A @ self.B)
return self.linear(x) + self.lora(x)
```

### 18.8 Instruction Collate

```python
new_item += [pad_token_id]
padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
inputs = torch.tensor(padded[:-1])
targets = torch.tensor(padded[1:])
targets[indices[1:]] = ignore_index
```

### 18.9 DPO

> ⚠ **시험 출제 포인트 외 — 참고용** (§18.9 / 아래 DPO 코드 빈칸)

```python
labels = labels[:, 1:].clone()
logits = logits[:, :-1, :]
log_probs = F.log_softmax(logits, dim=-1)
selected_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

model_logratios = model_chosen_logprobs - model_rejected_logprobs
reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
logits = model_logratios - reference_logratios
losses = -F.logsigmoid(beta * logits)
```

## 19. 예상문제 확장

### 19.1 개념형

1. GPT가 Transformer의 Decoder만 사용한다는 말의 의미를 설명하시오.
2. RNN이 Transformer보다 병렬화에 불리한 이유를 설명하시오.
3. Tokenizer가 필요한 이유를 “문자열과 Tensor” 관점에서 설명하시오.
4. BPE, WordPiece, Unigram의 차이를 비교하시오.
5. Special token과 padding token의 차이를 설명하시오.
6. token embedding과 positional embedding을 더하는 이유를 설명하시오.
7. RoPE가 상대적 위치 정보와 KV cache에 유리한 이유를 설명하시오.
8. Q, K, V를 하나의 projection으로 만들지 않고 분리하는 이유를 설명하시오.
9. Attention score를 `sqrt(d_k)`로 나누는 이유를 설명하시오.
10. Causal mask를 하지 않으면 GPT 학습에서 어떤 문제가 생기는지 설명하시오.
11. Multi-head attention이 여러 관점에서 문맥을 본다는 말을 예시로 설명하시오.
12. LayerNorm에서 scale과 shift가 학습 가능한 이유를 설명하시오.
13. Residual connection이 deep model 학습을 안정화하는 이유를 설명하시오.
14. GPT block의 입출력 차원이 같아야 하는 이유를 residual connection과 연결해 설명하시오.
15. *⚠ 참고용* Temperature가 generation 결과에 미치는 영향을 설명하시오.
16. Cross Entropy가 정답 token 확률과 어떻게 연결되는지 설명하시오.
17. train/validation/test split의 목적을 구분하시오.
18. Classification fine-tuning에서 output head를 교체하는 이유를 설명하시오.
19. *⚠ 참고용* Full fine-tuning과 LoRA의 차이를 설명하시오.
20. *⚠ 참고용* LoRA에서 A는 random, B는 zero로 초기화하는 이유를 설명하시오.
21. *⚠ 참고용* QLoRA가 GPU 메모리 문제를 어떻게 완화하는지 설명하시오.
22. Instruction fine-tuning 데이터 포맷을 예시로 쓰시오.
23. Instruction collate에서 padding target을 `-100`으로 바꾸는 이유를 설명하시오.
24. *⚠ 참고용* DPO가 reward model 없이 선호를 학습한다는 말의 의미를 설명하시오.
25. *⚠ 참고용* DPO에서 policy model과 reference model의 역할을 구분하시오.
26. *⚠ 참고용* DPO에서 prompt 부분을 mask하는 이유를 설명하시오.
27. RAG의 전체 pipeline을 순서대로 쓰시오.
28. RAG와 fine-tuning의 차이를 비교하시오.
29. *⚠ 참고용* Flash Attention이 attention approximation이 아니라 I/O 최적화라는 점을 설명하시오.
30. Quantization, Pruning, Distillation을 각각 한 문장으로 정의하시오.

### 19.2 코드형

1. sliding window 방식으로 input/target chunk를 생성하는 코드를 완성하시오.
2. CausalAttention에서 Q/K/V projection layer를 정의하시오.
3. Attention score 계산과 mask 적용 코드를 완성하시오.
4. Multi-head attention에서 `[B, T, D]`를 `[B, H, T, head_dim]`으로 바꾸는 코드를 완성하시오.
5. LayerNorm의 평균, 분산, scale, shift 계산 코드를 완성하시오.
6. TransformerBlock의 residual add 위치를 완성하시오.
7. GPTModel에서 token embedding, positional embedding, output head를 완성하시오.
8. generation loop에서 마지막 logits 선택과 `argmax`, `cat`을 완성하시오.
9. *⚠ 참고용* LoRALayer의 A/B shape와 forward 식을 완성하시오.
10. *⚠ 참고용* DPO loss의 log ratio와 `logsigmoid` 식을 완성하시오.

## 20. 최종 암기표

### 20.1 ◎ 시험 출제 포인트 직격

| 주제 | 반드시 기억할 것 |
|---|---|
| Tokenizer | text <-> token id, BPE는 빈도 높은 인접 쌍 병합 |
| Dataset | input은 현재 token들, target은 한 칸 뒤 token들 |
| Embedding | token embedding + positional embedding |
| GPT input | `[B, T] -> [B, T, D]` |
| GPT output | `[B, T, V]` |
| Attention | QK dot product -> scale -> mask -> softmax -> weighted sum of V |
| Mask | 미래 token 확률을 0으로 만들기 위해 `-inf` 적용 |
| Multi-head | 여러 head가 다른 관점으로 문맥 파악 |
| LayerNorm | feature 차원 정규화, scale/shift 학습 |
| Residual | 변화량 학습, deep network 안정화 |
| FFN | `D -> 4D -> D`, GELU |
| Generation | 마지막 time step logits로 다음 token 선택 |
| CE Loss | 정답 token 확률을 높이도록 학습 |
| Instruction | Instruction/Input/Response 포맷 (데이터 구성 의도) |
| RAG | retrieve한 context를 prompt에 넣어 generation |
| Quantization | 낮은 bit 이산값으로 표현 |
| Pruning | 중요도 낮은 구조 제거 |
| Distillation | teacher 지식을 student로 전달 |
| Activation 3종 | Sigmoid `[0,1]`, ReLU `[0,∞)` dead neuron, GELU smooth |
| Loss 3종 | MSE 회귀 / BCE 이진 / CE 다중·LM |
| Quant/Dequant 수식 | `q = round(x/s) + z`, `x ≈ s·(q − z)` |
| Pruning 기준 3종 | magnitude `|w|`, gradient `|w·∂L/∂w|`, head importance |
| Distill loss | `α · CE_hard + (1−α) · T² · KL(p_t‖p_s)` |

### 20.2 ⚠ 시험 출제 포인트 외 — 참고용

> 아래 항목들은 26.5월 시험 출제 포인트 PDF 키워드에 등장하지 않는다. 강의 자료(커리큘럼)에는 있으나 시험 대비는 위 ◎ 영역을 우선하라.

| 주제 | 한 줄 |
|---|---|
| LoRA | 기존 weight freeze, low-rank A/B만 학습 |
| QLoRA | 4-bit 양자화 + LoRA로 메모리 절감 |
| DPO | Prompt/Chosen/Rejected, policy vs reference |
| RLHF → DPO | reward model + PPO → 한 logsigmoid loss 로 축약 |
| Optimizer 계보 | SGD → Momentum → RMSprop → Adam (1·2차 모멘트) |
| Top-K + Temp | 상위 K 만 남기고 `softmax(logits/T)` 후 sample |
| Flash Attention | attention 수식 근사가 아니라 memory I/O 최적화 |
| Sliding Window | 일정 길이의 앞 token 만 본다 |
| Gated DeltaNet | Decay / Update / Output 3 gate |
| KV cache | 이전 K/V 재사용으로 추론 가속 |
| GQA / MLA | K,V head 수 ↓ / 차원 ↓ — KV cache 절감 |
| MoE | router 가 expert 일부만 활성화 (sparse) |

## 21. 시험 직전 체크리스트

### 21.1 ◎ 시험 출제 포인트 직격

- `[B, T] -> [B, T, D] -> [B, T, V]` 흐름을 설명할 수 있는가?
- input과 target이 왜 한 칸 shift되는지 예시로 말할 수 있는가?
- Q/K/V의 역할과 attention 수식을 코드로 쓸 수 있는가?
- causal mask가 softmax 이후 확률 0을 만든다는 점을 설명할 수 있는가?
- Multi-head attention shape 변환을 따라갈 수 있는가?
- LayerNorm, residual, FFN, output head의 역할을 구분할 수 있는가?
- Cross Entropy와 정답 token 확률의 관계를 설명할 수 있는가?
- Instruction fine-tuning 데이터 포맷·padding mask 의 *데이터 구성 의도* 를 설명할 수 있는가?
- RAG pipeline을 순서대로 쓰고 fine-tuning과 비교할 수 있는가?
- Quantization, Pruning, Distillation의 차이를 설명할 수 있는가?
- **★ Tensor 흐름**: `[B,T]→[B,T,D]→...→[B,T,V]→[B,V]→[B,1]` 의 모든 단계를 모듈 이름과 함께 일관되게 트레이스할 수 있는가?
- **★ sqrt(d_k) scaling** 이 점곱 분산을 안정화한다는 직관을 두 문장으로 설명할 수 있는가?
- **★ Multi-head shape 흐름** `view → transpose → softmax → transpose → contiguous → view` 를 순서대로 쓸 수 있는가?
- **★ 데이터 구성 의도**: pretraining shift, classification 마지막 token, instruction prompt mask 의 *이유* 를 각각 한 문장으로 말할 수 있는가?
- **★ Quantize / Dequantize 수식** `q = round(x/s) + z`, `x ≈ s·(q − z)` 를 외워 쓸 수 있는가?
- **★ Pruning 중요도 기준 3종** (magnitude / gradient / head importance) 과 structured vs unstructured 차이를 말할 수 있는가?
- **★ Distillation loss 수식** `loss = α · CE_hard + (1 − α) · T² · KL_soft` 와 *왜 soft label 이 더 풍부한가* 를 한 문장으로 설명할 수 있는가?
- llama-index 의 `VectorStoreIndex → as_retriever → as_query_engine` 흐름과 MCP 의 tool/resource 노출 개념을 RAG 와 연결할 수 있는가?

### 21.2 ⚠ 시험 출제 포인트 외 — 참고용

> 아래 항목들은 26.5월 시험 출제 포인트 PDF 키워드에 등장하지 않는다. 시간이 남을 때 보조 학습용으로만 활용하라.

- LoRA의 A/B shape, 초기화, freeze 전략을 말할 수 있는가?
- DPO 의 데이터 구조와 policy vs reference model 의 역할을 설명할 수 있는가?
- RLHF (reward model + PPO) 가 무엇이고 DPO 가 이를 어떻게 단순화했는지 한 문장으로 말할 수 있는가?
- Flash Attention, Sliding Window Attention, Gated DeltaNet의 목적을 구분할 수 있는가?
- GQA / MLA / MoE 가 각각 무엇을 어떻게 줄이는지 한 줄씩 답할 수 있는가?
- Top-K + Temperature 샘플링 코드의 핵심 4단계 (마지막 step slice → topk 외 -inf → `/T` → multinomial) 를 외울 수 있는가?

## 22. 웹 자료 기반 보강 설명

이 섹션은 논문과 공식 문서를 바탕으로 “왜 그렇게 동작하는지”를 더 쉽게 이해하기 위한 보강 설명이다. 기존 강의 PDF 내용과 연결해서 보면 시험 서술형 답안을 쓰기 좋다.

### 22.1 Transformer를 쉽게 이해하기

Transformer 논문인 *Attention Is All You Need*의 핵심은 RNN/CNN 없이 attention만으로 sequence transduction을 처리할 수 있다는 점이다. 시험에서는 논문 전체보다 다음 세 가지를 기억하면 충분하다.

1. RNN은 token을 순서대로 처리하므로 병렬화가 어렵다.
2. Transformer는 sequence 안의 모든 token 관계를 attention score 행렬로 한 번에 계산할 수 있다.
3. Decoder 쪽에서는 미래 token을 보면 안 되므로 mask를 적용한다.

직관:

```text
RNN: 앞 token을 처리해야 다음 token 처리 가능
Transformer: 모든 token 쌍의 관계를 행렬로 동시에 계산
GPT: Transformer decoder를 사용하되, 미래 token은 mask로 차단
```

Attention을 “검색”으로 비유하면 이해가 쉽다.

- Query: 내가 지금 찾고 싶은 질문
- Key: 각 문서 또는 token이 가진 색인
- Value: 실제로 가져올 내용

즉 attention은 “현재 token의 query로 모든 token의 key를 검색하고, 잘 맞는 token의 value를 많이 가져오는 과정”이다.

### 22.2 `sqrt(d_k)` scaling을 더 쉽게 이해하기

Query와 Key의 차원이 커지면 dot product 값이 커지는 경향이 있다. softmax는 입력값 차이가 커질수록 한 곳에 확률을 몰아준다. 그래서 scaling이 없으면 학습 초기에 attention이 너무 확정적으로 변해 gradient가 불안정해질 수 있다.

```text
차원 커짐 -> dot product 커짐 -> softmax가 뾰족해짐 -> 학습 불안정
해결: dot product를 sqrt(d_k)로 나눠 scale 조정
```

시험 답안 예:

> `sqrt(d_k)`로 나누는 이유는 QK 내적값의 분산이 key/query 차원에 따라 커지는 것을 완화해 softmax가 과도하게 포화되는 것을 막고 학습을 안정화하기 위해서다.

### 22.3 PyTorch `Embedding` 관점으로 이해하기

PyTorch 공식 문서에서 `nn.Embedding`은 “고정된 dictionary 크기와 embedding 크기를 가진 lookup table”로 설명된다. 즉 token id를 입력하면 해당 row의 embedding vector를 꺼내오는 구조다.

```python
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
out = embedding(input_ids)
```

shape 직관:

```text
input_ids shape = [B, T]
embedding table shape = [V, D]
output shape = [B, T, D]
```

중요한 점:

- token id 자체는 의미가 있는 숫자가 아니다.
- embedding table의 각 row가 학습되면서 의미 벡터가 된다.
- `padding_idx`가 지정되면 해당 padding row는 gradient 업데이트에서 제외될 수 있다.

시험에서 “Embedding은 Linear와 같은가?”라는 식으로 나오면:

- embedding은 one-hot vector에 weight matrix를 곱하는 것과 수학적으로 비슷하게 볼 수 있다.
- 하지만 실제 구현은 one-hot을 만들지 않고 index lookup으로 빠르게 처리한다.

### 22.4 PyTorch `LayerNorm` 관점으로 이해하기

PyTorch 공식 문서 기준으로 LayerNorm은 입력의 마지막 `normalized_shape` 차원들에 대해 평균과 분산을 계산한다. LLM에서 hidden shape가 `[B, T, D]`라면 보통 마지막 차원 `D`에 대해 정규화한다.

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (x - mean) / torch.sqrt(var + eps)
out = gamma * norm_x + beta
```

BatchNorm과 헷갈리지 말 것:

| 구분 | BatchNorm | LayerNorm |
|---|---|---|
| 통계 계산 축 | batch 방향 중심 | feature 방향 중심 |
| NLP/LLM 적합성 | sequence 길이와 batch 크기에 민감 | token별 hidden vector 정규화에 적합 |
| 학습/평가 차이 | running statistics 사용 가능 | 보통 입력 자체 통계 사용 |

시험 답안 예:

> LLM에서는 batch 크기나 sequence 길이가 유동적이므로 각 token의 hidden feature 차원에서 정규화하는 LayerNorm이 안정적이다. scale과 shift를 학습해 정규화로 잃을 수 있는 표현력을 회복한다.

### 22.5 PyTorch `CrossEntropyLoss`와 `ignore_index`

PyTorch 공식 문서 기준으로 `CrossEntropyLoss`는 input logits와 target 사이의 cross entropy를 계산한다. target이 class index일 때 `ignore_index`로 지정된 target 값은 loss와 gradient에 기여하지 않는다.

Instruction fine-tuning에서 padding target을 `-100`으로 바꾸는 이유:

```text
padding은 정답으로 맞혀야 할 실제 언어 token이 아님
-> loss에 포함하면 모델이 padding 예측을 학습하게 됨
-> ignore_index=-100으로 바꿔 loss 계산에서 제외
```

LLM 학습에서 자주 쓰는 shape 변환:

```python
# logits: [B, T, V]
# targets: [B, T]
loss = F.cross_entropy(
    logits.flatten(0, 1),   # [B*T, V]
    targets.flatten(),      # [B*T]
    ignore_index=-100
)
```

시험 포인트:

- CrossEntropyLoss는 raw logits를 입력으로 받는다.
- 내부적으로 log-softmax + negative log likelihood 형태로 이해할 수 있다.
- padding이나 prompt mask 영역은 target을 `-100`으로 두거나 별도 mask로 제외한다.

### 22.6 Tokenizer를 더 쉽게 이해하기

OpenAI의 `tiktoken`은 OpenAI 모델에 쓰이는 빠른 BPE tokenizer다. Hugging Face 문서에서도 주요 tokenizer 유형으로 BPE, WordPiece, Unigram을 비교한다.

세 tokenizer의 핵심 차이:

| 방식 | 쉬운 설명 | 대표 기억 포인트 |
|---|---|---|
| BPE | 자주 붙어 나오는 조각을 계속 합친다 | 빈도 높은 pair merge |
| WordPiece | 함께 나올 확률이 높은 조합을 고른다 | likelihood 기반 subword |
| Unigram | 큰 후보 집합에서 불필요한 token을 제거한다 | 여러 tokenization 후보 가능 |

BPE가 필요한 이유:

- 단어 단위 tokenizer는 미등록 단어, 오탈자, 복합어에 약하다.
- 문자 단위 tokenizer는 sequence가 너무 길어진다.
- subword tokenizer는 둘 사이의 절충이다.

시험 답안 예:

> BPE는 자주 등장하는 인접 문자 또는 subword pair를 반복적으로 병합해 vocabulary를 만든다. 그래서 흔한 단어는 긴 token으로 압축하고, 드문 단어는 작은 subword 조각으로 표현할 수 있다.

### 22.7 RAG를 더 쉽게 이해하기

RAG 논문은 parametric memory, 즉 모델 weight 안에 저장된 지식만 쓰는 방식의 한계를 보완하기 위해 retrieval을 결합한다. LlamaIndex 공식 문서도 RAG를 “LLM이 학습하지 않은 내 데이터를 index하고, query에 맞는 context를 골라 LLM에 전달하는 과정”으로 설명한다.

가장 쉬운 비유:

```text
Fine-tuning = 모델의 습관과 능력을 훈련한다.
RAG = 시험 중 참고자료를 찾아 옆에 펼쳐준다.
```

RAG가 좋은 경우:

- 사내 규정, 제품 문서, FAQ처럼 자주 바뀌는 지식
- 모델이 학습하지 않은 private data
- 근거 문서와 함께 답변해야 하는 업무

Fine-tuning이 좋은 경우:

- 특정 답변 스타일
- 분류, 추출, 포맷 변환 같은 반복 task
- 도메인 표현 방식이나 응답 형식을 몸에 익히게 할 때

RAG pipeline 이해:

```text
Loading: 원본 문서 불러오기
Chunking: 문서를 검색 가능한 조각으로 나누기
Indexing: embedding 후 vector store에 저장
Retrieval: query와 가까운 chunk 찾기
Generation: retrieved context와 query로 답변 생성
Evaluation: 답변 정확성, 충실성, 속도 평가
```

RAG에서 자주 틀리는 지점:

- chunk가 너무 작으면 답변에 필요한 문맥이 빠진다.
- chunk가 너무 크면 관련 없는 내용이 섞이고 context window를 낭비한다.
- embedding model이 약하면 좋은 chunk를 못 찾는다.
- top-k가 너무 작으면 근거가 부족하고, 너무 크면 noise가 많아진다.

### 22.8 MCP를 RAG와 연결해서 이해하기

MCP 공식 문서는 MCP를 AI 애플리케이션이 외부 시스템에 연결되는 표준으로 설명한다. RAG가 “검색해서 context를 넣는 패턴”이라면, MCP는 “모델/앱이 외부 데이터, 도구, 워크플로에 연결되는 표준 인터페이스”에 가깝다.

구분:

| 구분 | RAG | MCP |
|---|---|---|
| 목적 | 외부 지식 검색 후 답변 생성 | AI 앱과 외부 시스템 연결 표준화 |
| 핵심 요소 | index, retriever, context, generation | client, server, tools, resources |
| 예시 | 사내 PDF 검색 후 답변 | 파일 시스템, DB, 캘린더, 검색 도구 연결 |

시험 답안 예:

> RAG는 외부 문서를 검색해 LLM context로 넣는 아키텍처이고, MCP는 AI 애플리케이션이 외부 도구나 데이터 소스와 표준화된 방식으로 연결되게 하는 프로토콜이다. MCP 서버가 문서 검색 도구를 제공하면 RAG pipeline의 일부로 활용될 수 있다.

### 22.9 LoRA를 논문 관점으로 이해하기

> ⚠ **시험 출제 포인트 외 — 참고용** (§22.9 ~ §22.12 의 LoRA / QLoRA / DPO / Flash Attention 모두 동일)

LoRA 논문의 핵심 가정은 “fine-tuning으로 필요한 weight 변화량은 전체 차원을 다 쓰지 않고 낮은 rank로도 충분히 표현될 수 있다”는 것이다.

Full fine-tuning:

```text
W 전체를 업데이트
파라미터 수 = in_dim * out_dim
```

LoRA:

```text
W는 freeze
Delta W = A @ B만 학습
파라미터 수 = in_dim * r + r * out_dim
```

예:

```text
in_dim = 4096, out_dim = 4096, rank = 8
Full update 파라미터: 4096 * 4096 = 약 1,678만
LoRA 파라미터: 4096 * 8 + 8 * 4096 = 약 6.5만
```

이 차이가 LoRA의 메모리 효율을 만든다.

시험 답안 예:

> LoRA는 pretrained weight를 고정하고, task 적응에 필요한 변화량을 low-rank 행렬 A와 B의 곱으로 표현한다. rank가 작으면 학습 파라미터가 크게 줄어 full fine-tuning보다 메모리와 저장 비용이 낮다.

### 22.10 QLoRA를 더 쉽게 이해하기

QLoRA는 LoRA에 quantization을 결합한 방식이다. QLoRA 논문은 4-bit NormalFloat, double quantization, paged optimizer 같은 기법으로 메모리를 줄이면서 fine-tuning을 가능하게 한다.

직관:

```text
LoRA: 큰 모델은 그대로 두고 작은 adapter만 학습
QLoRA: 큰 모델을 4-bit로 저장하고 작은 adapter만 학습
```

주의:

- quantized weight는 저장 메모리를 줄인다.
- 계산 중에는 필요한 형식으로 dequantization될 수 있다.
- “메모리가 줄었다 = 무조건 빠르다”는 아니다. kernel, hardware, dequantization 비용이 중요하다.

시험 답안 예:

> QLoRA는 base LLM을 저비트로 양자화해 저장 메모리를 줄이고, LoRA adapter만 학습하는 방식이다. 따라서 큰 모델을 제한된 GPU 메모리에서도 fine-tuning할 수 있다.

### 22.11 DPO를 더 쉽게 이해하기

DPO 논문의 핵심은 preference optimization을 별도의 reward model과 RL 없이 supervised learning처럼 직접 풀 수 있다는 점이다.

비유:

```text
SFT: 좋은 답변 예시를 따라 쓰게 함
DPO: 두 답변 중 어떤 답이 더 좋은지 비교해 선호 방향을 학습
```

DPO가 보는 데이터:

```text
같은 prompt에 대해
chosen response: 더 선호되는 답
rejected response: 덜 선호되는 답
```

핵심 수식 직관:

```text
policy가 chosen에 준 점수 - policy가 rejected에 준 점수
를
reference가 chosen/rejected에 준 점수 차이보다 크게 만들기
```

즉 DPO는 “기준 모델 대비, 학습 모델이 chosen을 rejected보다 더 선호하게” 만든다.

시험 답안 예:

> DPO는 policy model과 reference model의 chosen/rejected 로그확률 차이를 비교한다. policy가 reference보다 chosen과 rejected를 더 잘 구분하도록 `-logsigmoid(beta * logits)` 형태의 loss를 최소화한다.

### 22.12 FlashAttention을 더 쉽게 이해하기

FlashAttention 논문의 핵심은 “attention 계산량 자체보다 GPU 메모리 읽기/쓰기가 병목이 될 수 있다”는 관점이다. FlashAttention은 attention을 근사하지 않고 정확한 attention을 계산하되, tiling과 SRAM 활용으로 HBM 접근을 줄인다.

비유:

```text
일반 attention:
큰 책 전체를 매번 창고에서 꺼내 읽고 다시 넣음

FlashAttention:
필요한 페이지 묶음만 책상 위 빠른 공간에 올려두고 계산
```

핵심 키워드:

- exact attention: 근사가 아니다.
- IO-aware: GPU memory hierarchy를 고려한다.
- HBM: 크지만 상대적으로 느린 GPU 메모리
- SRAM: 작지만 빠른 on-chip memory
- tiling: Q/K/V block을 나누어 계산
- online softmax: 전체 score matrix를 저장하지 않고 softmax에 필요한 통계를 누적

시험 답안 예:

> FlashAttention은 attention 수식을 바꾸거나 sparse approximation을 쓰는 것이 아니라, tiling과 online softmax로 HBM과 SRAM 사이의 메모리 I/O를 줄여 정확한 attention을 더 빠르고 메모리 효율적으로 계산하는 방법이다.

### 22.13 Quantization을 더 쉽게 이해하기

PyTorch 문서와 블로그에서는 quantization을 FP32 모델 대비 INT8 등을 사용해 모델 크기와 메모리 bandwidth 요구를 줄이는 방식으로 설명한다.

직관:

```text
FP32: 숫자 하나를 32비트로 저장
INT8: 숫자 하나를 8비트로 저장
이론적으로 weight 저장량은 약 1/4
```

기본 개념:

- scale: 정수와 실수 사이 비율
- zero-point: 실수 0을 정수 공간에서 어디로 둘지
- quantization: float -> int
- dequantization: int -> float 근사값

주의할 점:

- quantization error가 생긴다.
- 작은 모델이나 민감한 layer는 성능 저하가 클 수 있다.
- 실제 속도 개선은 hardware와 quantized kernel 지원에 달려 있다.

### 22.14 Pruning을 더 쉽게 이해하기

PyTorch pruning tutorial은 module 안의 weight에 pruning mask를 적용하는 방식으로 pruning을 설명한다.

직관:

```text
중요하지 않은 연결을 0으로 만들거나 제거
-> 연산량 또는 저장량 감소 기대
```

종류:

- unstructured pruning: 개별 weight를 제거
- structured pruning: neuron, channel, head 같은 구조 단위 제거

시험에서는 structured pruning이 실제 속도 개선과 더 직접적으로 연결된다는 점을 기억하면 좋다. 단순히 weight 몇 개를 0으로 만드는 unstructured pruning은 sparse kernel이 없으면 속도 개선이 작을 수 있다.

### 22.15 Distillation을 더 쉽게 이해하기

Hinton 등의 knowledge distillation 논문은 큰 teacher 모델의 지식을 작은 student 모델에 전달하는 아이디어를 정리했다.

핵심:

- hard label: 정답 하나만 알려준다.
- soft label: teacher가 각 class에 준 확률분포를 알려준다.

예:

```text
정답: 고양이
hard label: 고양이=1, 나머지=0
soft label: 고양이=0.75, 호랑이=0.15, 강아지=0.05, ...
```

soft label에는 class 간 유사도 정보가 담겨 있다. 그래서 student는 정답뿐 아니라 teacher가 헷갈린 방식까지 배울 수 있다.

LLM distillation 예:

- 큰 모델이 만든 답변을 작은 모델의 학습 데이터로 사용
- teacher logits 또는 확률분포를 student가 따라가게 함
- reasoning trace나 instruction response를 작은 모델에 전달

## 23. 웹 자료 기반 헷갈리는 개념 비교

### 23.1 Pretraining, SFT, DPO, RAG 비교

> ⚠ **시험 출제 포인트 외 — 참고용** (DPO 항목 포함)

| 구분 | 무엇을 바꾸는가 | 데이터 형태 | 목적 |
|---|---|---|---|
| Pretraining | 전체 모델 weight | 대규모 raw text | 다음 token 예측 능력 학습 |
| SFT/Instruction tuning | 모델 weight | instruction-response | 지시 따르기 학습 |
| DPO | policy model weight | prompt/chosen/rejected | 선호되는 답변 방향 학습 |
| RAG | 보통 weight 안 바꿈 | 외부 문서 index | 검색 근거 기반 답변 |

### 23.2 LoRA, QLoRA, Quantization 비교

> ⚠ **시험 출제 포인트 외 — 참고용** (LoRA / QLoRA 만. Quantization 자체는 ◎)

| 구분 | 핵심 | 학습 여부 | 목적 |
|---|---|---|---|
| LoRA | low-rank adapter 학습 | adapter만 학습 | fine-tuning 비용 절감 |
| QLoRA | quantized base + LoRA | adapter만 학습 | GPU 메모리 절감 |
| Quantization | 숫자를 낮은 bit로 표현 | 보통 추론 최적화, QAT도 가능 | 모델 크기/메모리/latency 절감 |

### 23.3 Attention 최적화 비교

> ⚠ **시험 출제 포인트 외 — 참고용** (Flash Attention / Sliding Window / KV cache 등)

| 방식 | 핵심 아이디어 | 장점 | 주의 |
|---|---|---|---|
| Full Attention | 모든 token pair 계산 | 가장 직접적 | O(T^2) 메모리/계산 |
| Causal Attention | 미래 token mask | GPT 학습에 필수 | 과거만 참조 |
| Sliding Window | 가까운 과거 일부만 참조 | 긴 sequence 비용 감소 | 먼 문맥 손실 |
| FlashAttention | I/O-aware exact attention | 빠르고 메모리 효율적 | 수식 근사가 아님 |

## 24. 참고한 웹 자료

- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- BPE/subword: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), [OpenAI tiktoken](https://github.com/openai/tiktoken), [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/v4.42.0/tokenizer_summary)
- PyTorch: [Embedding](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Embedding.html), [LayerNorm](https://docs.pytorch.org/docs/2.11/generated/torch.nn.LayerNorm.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html)
- RAG: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), [LlamaIndex Introduction to RAG](https://docs.llamaindex.ai/en/stable/understanding/rag/), [LlamaIndex Retriever](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/)
- MCP: [Model Context Protocol introduction](https://modelcontextprotocol.io/docs/getting-started/intro)
- LoRA/QLoRA: [LoRA paper](https://arxiv.org/abs/2106.09685), [Hugging Face PEFT LoRA docs](https://huggingface.co/docs/peft/main/package_reference/lora), [QLoRA paper](https://arxiv.org/abs/2305.14314)
- DPO: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- FlashAttention: [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- Quantization/Pruning/Distillation: [PyTorch Quantization](https://docs.pytorch.org/docs/stable/quantization), [PyTorch Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html), [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
