"""
【EOPD 核心】On-Policy Distillation 训练器。

本文件实现 OnPolicyDistillTrainer 类，继承自 RayPPOTrainer，
是 EOPD 算法的训练入口。核心功能：

1. 将教师模型注册为 Role.RefPolicy（参考策略角色）
2. 在计算教师 log_prob 前执行 EOS token 替换（适配不同 tokenizer）
3. 强制 use_reference_policy=True，确保教师 log_prob 被计算

数据流：
- 学生模型 (actor) 生成 rollout 数据 → 计算学生 log_prob
- 教师模型 (ref_policy) 对相同数据计算 teacher log_prob、entropy、top-k logits
- 两者一起传入 compute_policy_loss_on_policy_distill 计算损失

与其他模块的关系：
- 继承 RayPPOTrainer（ray_trainer.py）的分布式训练框架
- 使用 dp_actor.py 中的 DataParallelPPOActor 执行前向/反向传播
- 依赖 core_algos.py 中的 compute_policy_loss_on_policy_distill 损失函数
"""

from omegaconf import OmegaConf

from verl.single_controller.ray import RayClassWithInitArgs
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.utils.fs import copy_to_local
from verl import DataProto


class OnPolicyDistillTrainer(RayPPOTrainer):
    """【EOPD 核心】On-Policy Distillation 训练器。

    继承 RayPPOTrainer，主要改动：
    1. init_workers(): 将教师模型初始化为 ref_policy_wg（参考策略 worker group）
    2. _compute_ref_log_prob(): 用教师模型计算 log_prob、entropy、top-k logits
    3. _replace_eos_token_for_teacher(): 适配学生/教师 tokenizer 差异

    训练流程中，每个 step：
    - 学生模型生成 rollout → 计算 log_prob（带梯度）
    - 教师模型计算 ref_log_prob、ref_entropy、ref_topk（无梯度）
    - 两者输入 policy_loss_fn 计算 EOPD 损失并反向传播
    """

    def __init__(self, *args, **kwargs):
        """Initialize the on-policy distillation trainer.
        
        Args are identical to RayPPOTrainer.__init__()
        """
        super().__init__(*args, **kwargs)
        self.teacher_wg = None
        # Force use_reference_policy to True so that ref_log_prob is computed
        # This must be set in __init__ before fit() is called
        self.use_reference_policy = True
        
        # Add Role.RefPolicy to role_worker_mapping so that parent's init_workers() can create it
        # Teacher uses the same worker class as actor_rollout
        actor_role = Role.ActorRolloutRef if (
            self.config.algorithm.use_kl_in_reward or self.config.actor_rollout_ref.actor.use_kl_loss
        ) else Role.ActorRollout
        
        if Role.RefPolicy not in self.role_worker_mapping:
            # Use the same worker class as actor_rollout for teacher
            self.role_worker_mapping[Role.RefPolicy] = self.role_worker_mapping[actor_role]
        
        # Add Role.RefPolicy to role_worker_mapping so that parent's init_workers() can create it
        # Teacher uses the same worker class as actor_rollout
        actor_role = Role.ActorRolloutRef if (
            self.config.algorithm.use_kl_in_reward or self.config.actor_rollout_ref.actor.use_kl_loss
        ) else Role.ActorRollout
        
        if Role.RefPolicy not in self.role_worker_mapping:
            # Use the same worker class as actor_rollout for teacher
            self.role_worker_mapping[Role.RefPolicy] = self.role_worker_mapping[actor_role]

    def init_workers(self):
        """Initialize workers including teacher model worker group as ref model."""
        # Check if teacher model is configured
        teacher_model_path = self.config.actor_rollout_ref.get("teacher_model", {}).get("path", None)
        if teacher_model_path is None:
            raise ValueError(
                "teacher_model.path must be set in config.actor_rollout_ref.teacher_model.path "
                "for OnPolicyDistillTrainer"
            )
        
        # Ensure Role.RefPolicy is mapped to global_pool (needed for resource pool creation)
        if Role.RefPolicy not in self.resource_pool_manager.mapping:
            self.resource_pool_manager.mapping[Role.RefPolicy] = "global_pool"
        
        # Call parent's init_workers - it will detect teacher_model.path and use it for ref_policy
        super().init_workers()
        
        # Verify that teacher (ref_policy_wg) was successfully initialized
        if self.ref_policy_wg is None:
            raise RuntimeError(
                "Teacher model (ref_policy_wg) was not initialized. "
                "This should not happen in OnPolicyDistillTrainer. "
                "Check that teacher_model.path is correctly configured."
            )
        
        # Get the teacher worker group (already registered as Role.RefPolicy)
        self.teacher_wg = self.ref_policy_wg

    def _replace_eos_token_for_teacher(self, batch):
        """Replace <|endoftext|> (151643) with <|im_end|> (151645) in batch for teacher model.
        
        This function creates a copy of the batch and replaces the first occurrence of
        <|endoftext|> token (id 151643) with <|im_end|> token (id 151645) in responses only.
        This is needed because student model uses <|endoftext|> while teacher expects <|im_end|>.
        
        Note: Only responses are modified, not input_ids. The original batch is not modified,
        so student model can safely use the original batch for computing log probabilities.
        
        Args:
            batch: DataProto containing input_ids and responses
            
        Returns:
            DataProto: A copy of batch with token IDs replaced in responses only
        """
        import torch
        from verl import DataProto
        
        # Token IDs
        ENDOFTEXT_TOKEN_ID = 151643
        IM_END_TOKEN_ID = 151645
        
        # Clone the batch TensorDict and create a new DataProto
        # This ensures we don't modify the original batch
        if batch.batch is not None:
            # Clone the entire batch TensorDict
            batch_td_clone = batch.batch.clone()
        else:
            batch_td_clone = None
        
        # Only replace in responses: [batch_size, response_length]
        # input_ids is not modified as it doesn't contain the EOS token we need to replace
        if batch_td_clone is not None and "responses" in batch_td_clone:
            responses = batch_td_clone["responses"]
            # Store original shape for validation
            original_shape = responses.shape
            
            # For each sequence, find the first occurrence of ENDOFTEXT_TOKEN_ID and replace it
            for i in range(responses.shape[0]):
                # Find all positions where ENDOFTEXT_TOKEN_ID appears
                matches = (responses[i] == ENDOFTEXT_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    # Replace only the first occurrence
                    first_match_idx = matches[0].item()
                    
                    # Get 10 tokens before replacement (including the token to be replaced)
                    # [first_match_idx - 9, first_match_idx] (total 10 tokens)
                    start_idx_before = max(0, first_match_idx - 9)
                    tokens_before = responses[i, start_idx_before:first_match_idx + 1].cpu().clone()
                    
                    # Decode before replacement
                    tokens_before_list = tokens_before.tolist()
                    decoded_before = self.tokenizer.decode(tokens_before_list, skip_special_tokens=False)
                    
                    # Replace the token in responses
                    responses[i, first_match_idx] = IM_END_TOKEN_ID

                    # Update input_ids if available
                    # This is CRITICAL: models use input_ids for forward pass, not responses
                    if "input_ids" in batch_td_clone:
                        input_ids = batch_td_clone["input_ids"]
                        # Calculate index in input_ids
                        # Assuming input_ids ends with responses: input_ids = [..., responses]
                        # So the last len(responses) tokens of input_ids match responses
                        # Index mapping: idx_in_input = idx_in_response + (len(input) - len(response))
                        seq_len = input_ids.shape[1]
                        resp_len = responses.shape[1]
                        if seq_len >= resp_len:
                            offset = seq_len - resp_len
                            input_idx = offset + first_match_idx
                            
                            # Verify and replace
                            if input_ids[i, input_idx] == ENDOFTEXT_TOKEN_ID:
                                input_ids[i, input_idx] = IM_END_TOKEN_ID
                                # print(f"  [Input Update] Also replaced input_ids at index {input_idx} (offset {offset})")
                            else:
                                #print(f"  [Input Warning] input_ids[{input_idx}] ({input_ids[i, input_idx].item()}) != ENDOFTEXT_TOKEN_ID")
                                pass
                        else:
                             print(f"  [Input Error] input_ids length ({seq_len}) < responses length ({resp_len})")
                    
                    # Get 10 tokens after replacement (including the replaced token)
                    # [first_match_idx - 9, first_match_idx] (total 10 tokens, but token at first_match_idx is now replaced)
                    tokens_after = responses[i, start_idx_before:first_match_idx + 1].cpu().clone()
                    
                    # Decode after replacement
                    tokens_after_list = tokens_after.tolist()
                    decoded_after = self.tokenizer.decode(tokens_after_list, skip_special_tokens=False)
                    
                    # print(f"[_replace_eos_token_for_teacher] Sample {i}: Replacing <|endoftext|> (id {ENDOFTEXT_TOKEN_ID}) with <|im_end|> (id {IM_END_TOKEN_ID}) at index {first_match_idx}")
                    # print(f"  Before (10 tokens including replacement position): {decoded_before}")
                    # print(f"  After (10 tokens including replacement position): {decoded_after}")
                    # print(f"  Token IDs before: {tokens_before_list}")
                    # print(f"  Token IDs after: {tokens_after_list}")
                else:
                    # print(f"[_replace_eos_token_for_teacher] Sample {i}: No <|endoftext|> token found in responses")
                    pass
            
            # Verify that tensor shape hasn't changed after modification
            assert responses.shape == original_shape, (
                f"responses tensor shape changed after EOS token replacement: "
                f"original={original_shape}, after={responses.shape}"
            )
        
        # Create a new DataProto with the modified batch
        # Copy non_tensor_batch and meta_info (shallow copy is fine for dicts)
        batch_copy = DataProto(
            batch=batch_td_clone,
            non_tensor_batch=batch.non_tensor_batch.copy() if batch.non_tensor_batch else {},
            meta_info=batch.meta_info.copy() if batch.meta_info else {},
        )
        
        return batch_copy

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """Compute ref_log_prob using teacher model instead of default ref model.
        
        The teacher model is registered as Role.RefPolicy in init_workers(),
        so self.ref_policy_wg already points to the teacher model.
        
        Before computing log probabilities, replaces <|endoftext|> with <|im_end|>
        in responses for teacher model compatibility. The original batch is not modified,
        so student model can safely use the original batch.
        """
        # Verify teacher is initialized
        if self.ref_policy_wg is None or self.teacher_wg is None:
            raise RuntimeError(
                "Teacher model (ref_policy_wg) is not initialized. "
                "This should not happen in OnPolicyDistillTrainer. "
                "Ensure init_workers() was called successfully."
            )
        
        # Replace EOS token for teacher model (creates a copy, doesn't modify original batch)
        # This ensures teacher model receives <|im_end|> instead of <|endoftext|>
        batch_for_teacher = self._replace_eos_token_for_teacher(batch)
        
        # self.ref_policy_wg is already set to teacher_wg in init_workers()
        # Call parent's method with the modified batch (original batch is unchanged)
        return super()._compute_ref_log_prob(batch_for_teacher)
