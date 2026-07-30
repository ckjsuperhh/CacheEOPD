<!-- 中文摘要：本文档分析教师模型中 Top-K Logits 返回功能的实现，对比 FSDP 与 Megatron 后端 -->
# Teacher Model에서 Top-K Logits 반환 기능 분석

## 1. 현재 구현 상태

### 1.1 현재 `_compute_ref_log_prob`의 동작

현재 `_compute_ref_log_prob`는 **이미 생성된 토큰에 대한 log probability만 반환**합니다.

### 1.2 FSDP vs Megatron

**FSDP (DataParallelPPOActor) 사용 시:**
- ✅ **Logits 접근 가능**: `use_fused_kernels=False` (기본값)인 경우 `output.logits`를 직접 사용 가능
- ✅ **Top-k 추출 가능**: Logits가 있으므로 `torch.topk()`로 top-k 추출 가능
- ⚠️ **주의**: `use_fused_kernels=True`인 경우 `output.log_probs`만 반환되므로 logits 접근 불가

**Megatron 사용 시:**
- ✅ **Logits 접근 가능**: 동일하게 `use_fused_kernels=False`인 경우 가능
- ✅ **기존 구현 참고**: `recipe/gkd/`에 이미 구현된 예제 존재

**호출 흐름:**
```
ray_trainer.py:1600 (_compute_ref_log_prob 호출)
  → on_policy_distill_trainer.py:92 (super() 호출)
    → ray_trainer.py:1254-1273 (실제 실행)
      → ref_policy_wg.compute_ref_log_prob()
        → worker.compute_ref_log_prob()
          → ref_policy.compute_log_prob()
            → _forward_micro_batch() → logits 계산
            → logprobs_from_logits() → log_probs만 반환
```

**반환값:**
- `ref_log_prob`: Shape `(batch_size, response_length)` - 각 토큰 위치에서 실제 생성된 토큰에 대한 log probability만 포함

### 1.3 현재 구현의 한계

1. **Logits 정보 손실**: `_forward_micro_batch`에서 logits를 계산하지만, `logprobs_from_logits`를 통해 특정 토큰의 log probability만 추출하고 logits는 버려짐
2. **Soft Knowledge Distillation 불가**: Forward KL divergence를 사용한 soft distillation을 위해서는 전체 vocabulary에 대한 logits 또는 top-k logits가 필요
3. **현재는 Hard Distillation만 가능**: 현재 `on_policy_distill` loss는 teacher와 student의 log probability 차이만 사용 (reverse KL)

## 2. Top-K Logits 반환 기능 추가 가능성

### 2.1 가능성: ✅ **가능함**

다음 이유로 구현 가능:

1. **이미 logits 계산됨**: `_forward_micro_batch`에서 이미 logits를 계산하고 있음 (`output.logits`)
2. **기존 구현 참고 가능**: `recipe/gkd/`에 top-k logits를 사용하는 knowledge distillation 구현이 이미 존재
3. **메모리 효율적**: 전체 vocab logits 대신 top-k만 반환하면 메모리 사용량이 크게 줄어듦

### 2.2 기존 구현 참고: `recipe/gkd/`

`recipe/gkd/megatron_workers.py`와 `recipe/gkd/README.md`를 보면:
- Teacher가 top-k log-probabilities와 token indices를 반환
- Student는 sparse KL divergence를 계산하여 teacher의 top-k distribution과 매칭

## 3. 구현 방법

### 3.1 수정이 필요한 파일들

#### 3.1.1 Worker Level (`verl/workers/actor/dp_actor.py`)

**현재 (FSDP):**
```python
def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
    # ...
    # _forward_micro_batch 호출하여 log_probs만 반환
    # use_fused_kernels=False인 경우 logits는 계산되지만 버려짐
```

**수정 방향:**
1. `_forward_micro_batch`의 반환값에 logits 추가 (use_fused_kernels=False인 경우)
2. `compute_log_prob`에 `return_topk_logits` 파라미터 추가
3. Top-k logits와 indices를 계산하여 반환

**구체적 수정:**
```python
# _forward_micro_batch 수정
def _forward_micro_batch(
    self, micro_batch, temperature, calculate_entropy=False, return_logits=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # ...
    if self.use_fused_kernels:
        # logits 없음
        return entropy, log_probs, full_log_probs, None
    else:
        logits = output.logits  # 이미 계산됨
        logits.div_(temperature)
        # ...
        if return_logits:
            logits_resp = logits[:, -response_length - 1 : -1, :]
            return entropy, log_probs, full_log_probs, logits_resp
        return entropy, log_probs, full_log_probs, None
```

**예시 수정 (FSDP용):**
```python
def compute_log_prob(
    self, 
    data: DataProto, 
    calculate_entropy=False,
    return_topk_logits=False,
    topk=50
) -> tuple[torch.Tensor, ...]:
    # ...
    # _forward_micro_batch 호출
    entropy, log_probs, full_log_probs = self._forward_micro_batch(...)
    
    if return_topk_logits:
        # use_fused_kernels=False인 경우에만 가능
        if self.use_fused_kernels:
            raise ValueError("return_topk_logits requires use_fused_kernels=False")
        
        # _forward_micro_batch 내부에서 logits를 반환하도록 수정 필요
        # 또는 별도로 forward를 다시 호출하여 logits 획득
        # logits_resp = logits[:, -response_length - 1 : -1, :]  # (bsz, resp_len, vocab_size)
        # topk_logits, topk_indices = torch.topk(logits_resp, k=topk, dim=-1)
        # topk_log_probs = F.log_softmax(topk_logits, dim=-1)
        # return log_probs, (topk_log_probs, topk_indices)
    
    return log_probs, None
```

**주의사항:**
- `use_fused_kernels=True`인 경우 logits가 없으므로 top-k 추출 불가
- `use_fused_kernels=False` (기본값)인 경우에만 가능
- `_forward_micro_batch`에서 logits를 반환하도록 수정 필요

#### 3.1.2 Worker Wrapper (`verl/workers/fsdp_workers.py`, `verl/workers/megatron_workers.py`)

**수정:**
```python
def compute_ref_log_prob(self, data: DataProto):
    # ...
    output, topk_info = self.ref_policy.compute_log_prob(
        data=data, 
        calculate_entropy=False,
        return_topk_logits=True,  # 새 파라미터
        topk=self.config.ref.get("topk_logits", 50)  # config에서 가져오기
    )
    
    if topk_info is not None:
        topk_log_probs, topk_indices = topk_info
        output = DataProto.from_dict(tensors={
            "ref_log_prob": output,
            "ref_topk_log_probs": topk_log_probs,
            "ref_topk_indices": topk_indices
        })
    else:
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
```

#### 3.1.3 Trainer Level (`verl/trainer/ppo/ray_trainer.py`)

**수정:**
```python
def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
    # ...
    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    
    # topk 정보가 있으면 batch에 추가
    if "ref_topk_log_probs" in ref_log_prob.batch:
        # 이미 포함되어 있으므로 그대로 반환
        pass
    
    return ref_log_prob
```

#### 3.1.4 Loss Function (`verl/trainer/ppo/core_algos.py`)

**새로운 loss 함수 추가 또는 기존 함수 수정:**

```python
@register_policy_loss("on_policy_distill_soft")  # 새 loss type
def compute_policy_loss_on_policy_distill_soft(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    ref_topk_log_probs: torch.Tensor | None = None,  # 새 파라미터
    ref_topk_indices: torch.Tensor | None = None,     # 새 파라미터
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Soft knowledge distillation using forward KL divergence on top-k logits.
    
    Args:
        ref_topk_log_probs: Teacher's top-k log probabilities, shape (batch, seq_len, topk)
        ref_topk_indices: Teacher's top-k token indices, shape (batch, seq_len, topk)
    """
    # Student의 logits에서 teacher의 top-k 위치만 추출
    # Forward KL: KL(P_teacher || P_student) on top-k support
    # ...
```

### 3.2 Config 추가

**`verl/trainer/config/actor_config.py` 또는 관련 config 파일:**
```python
ref:
    log_prob_micro_batch_size_per_gpu: 16
    log_prob_max_token_len_per_gpu: 4096
    log_prob_use_dynamic_bsz: True
    topk_logits: 50  # 새 파라미터: top-k 개수
    return_topk_logits: True  # 새 파라미터: top-k 반환 여부
```

## 4. 메모리 및 성능 고려사항

### 4.1 메모리 사용량

**현재 (log_probs만):**
- Shape: `(batch_size, response_length)`
- 예: `(128, 4096)` = 512KB (float32)

**Top-k logits 추가 시:**
- `topk_log_probs`: `(batch_size, response_length, topk)` = `(128, 4096, 50)` = 100MB (float32)
- `topk_indices`: `(batch_size, response_length, topk)` = `(128, 4096, 50)` = 100MB (int64)
- **총 추가 메모리**: ~200MB per batch

**최적화 방안:**
1. Top-k를 작게 설정 (예: 20-50)
2. Response length가 긴 경우 chunking
3. CPU offload 고려

### 4.2 계산 오버헤드

**추가 연산:**
1. `torch.topk()`: O(vocab_size * log(topk)) per token
2. `F.log_softmax()`: O(topk) per token

**예상 오버헤드:**
- Top-k 추출: ~5-10% 추가 시간
- 전체 forward pass 대비 상대적으로 작음

## 5. 구현 단계별 계획

### Phase 1: Worker Level 수정
1. `dp_actor.py`의 `compute_log_prob`에 `return_topk_logits` 파라미터 추가
2. `_forward_micro_batch`에서 logits 보존 및 top-k 추출 로직 추가
3. 반환값에 top-k 정보 포함

### Phase 2: Worker Wrapper 수정
1. `fsdp_workers.py`의 `compute_ref_log_prob` 수정
2. `megatron_workers.py`의 `compute_ref_log_prob` 수정 (필요시)
3. Config에서 top-k 파라미터 읽기

### Phase 3: Trainer Level 수정
1. `ray_trainer.py`의 `_compute_ref_log_prob` 수정 (이미 top-k 정보가 포함되어 있으면 그대로 전달)
2. `on_policy_distill_trainer.py`는 수정 불필요 (부모 클래스 사용)

### Phase 4: Loss Function 추가
1. `core_algos.py`에 새로운 loss 함수 추가: `compute_policy_loss_on_policy_distill_soft`
2. Forward KL divergence 구현
3. Config에 새 loss type 등록

### Phase 5: 테스트 및 검증
1. 단위 테스트 작성
2. 메모리 사용량 확인
3. 학습 성능 비교 (hard vs soft distillation)

## 6. 참고 구현

### 6.1 `recipe/gkd/` 구현 참고

- **파일**: `recipe/gkd/megatron_workers.py`
- **핵심 로직**: `vocab_parallel_kl_divergence` 함수
- **특징**: 
  - Top-k log-probabilities와 indices를 teacher에서 반환
  - Sparse KL divergence 계산 (top-k support만 사용)
  - 메모리 효율적인 구현

### 6.2 기존 코드 재사용 가능 부분

1. **Top-k 추출**: `torch.topk()` 사용
2. **Log softmax**: `F.log_softmax()` 사용
3. **KL divergence**: `verl/utils/torch_functional.py`의 기존 함수 활용 가능

## 7. 결론

### 7.1 가능성
✅ **구현 가능**: 현재 구조에서 logits 정보를 보존하고 top-k만 추출하여 반환하는 것이 가능함

### 7.2 장점
1. **Soft Knowledge Distillation 지원**: Forward KL을 사용한 더 부드러운 distillation 가능
2. **기존 구현 참고 가능**: `recipe/gkd/`의 구현을 참고하여 빠르게 구현 가능
3. **메모리 효율적**: 전체 vocab 대신 top-k만 사용

### 7.3 주의사항
1. **메모리 사용량 증가**: Batch size와 response length에 따라 메모리 사용량이 증가할 수 있음
2. **Config 호환성**: 기존 config와의 호환성 유지 필요
3. **use_fused_kernels 제약**: 
   - ✅ `use_fused_kernels=False` (기본값): Logits 접근 가능, top-k 추출 가능
   - ❌ `use_fused_kernels=True`: Logits 없음, top-k 추출 불가
4. **FSDP 구현**: 현재 FSDP (DataParallelPPOActor)에서도 logits 접근 가능하므로 구현 가능
5. **다양한 Worker 타입**: FSDP, Megatron, Engine 등 다양한 worker 타입에 대한 지원 필요

### 7.4 다음 단계
1. Phase 1부터 단계적으로 구현 시작
2. `recipe/gkd/` 구현을 참고하여 빠르게 프로토타입 작성
3. 작은 규모로 테스트 후 점진적으로 확장

