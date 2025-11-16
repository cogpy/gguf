# TinyTransformer - Symbolic Representation

## Model Architecture

- **model_name**: TinyTransformer
- **architecture**: transformer
- **representation**: symbolic
- **vocab_size**: 10
- **embedding_dim**: 5
- **context_length**: 5
- **num_blocks**: 1
- **num_heads**: 1

## Notation

- `x`: Input token IDs, x ∈ {0,1,...,9}^L where L=5
- `E`: Embedding matrix (vocab_size × d_model)
- `h`: Hidden state / embeddings
- `Q, K, V`: Query, Key, Value matrices
- `W^Q, W^K, W^V`: Query, Key, Value projection weights
- `W^O`: Attention output projection
- `A`: Attention weights (after softmax)
- `W^{ff}_1, W^{ff}_2`: Feed-forward network weights
- `W^{out}`: Output projection to vocabulary
- `LN`: Layer normalization
- `d_k`: Dimension of keys (= d_model / num_heads = 5)
- `L`: Sequence length (= 5)
- `d_{model}`: Model dimension (= 5)
- `d_{ff}`: Feed-forward dimension (= 5)
- `V`: Vocabulary size (= 10)

## Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| E | 10 × 5 | float32 | Token embedding matrix |
| W^Q | 5 × 5 | float32 | Query projection weights |
| W^K | 5 × 5 | float32 | Key projection weights |
| W^V | 5 × 5 | float32 | Value projection weights |
| W^O | 5 × 5 | float32 | Attention output projection |
| \gamma_1 | 5 | float32 | Layer norm 1 scale |
| \beta_1 | 5 | float32 | Layer norm 1 bias |
| W^{ff}_1 | 5 × 5 | float32 | FFN first layer weights |
| W^{ff}_2 | 5 × 5 | float32 | FFN second layer weights |
| W^{out} | 5 × 10 | float32 | Output projection to vocabulary |

## Forward Pass Equations

1. **h_0** = `E[x]`
   - Embedding lookup
   - Shape: (batch, L, d_model)

2. **Q** = `h_0 W^Q`
   - Query projection
   - Shape: (batch, L, d_k)

3. **K** = `h_0 W^K`
   - Key projection
   - Shape: (batch, L, d_k)

4. **V** = `h_0 W^V`
   - Value projection
   - Shape: (batch, L, d_k)

5. **S** = `QK^T / sqrt(d_k)`
   - Scaled attention scores
   - Shape: (batch, L, L)

6. **A** = `softmax(S)`
   - Attention weights
   - Shape: (batch, L, L)

7. **h_attn** = `(AV)W^O`
   - Attention output
   - Shape: (batch, L, d_model)

8. **h_1** = `LN(h_0 + h_attn; gamma_1, beta_1)`
   - Post-attention with residual and layer norm
   - Shape: (batch, L, d_model)

9. **h_ff** = `GELU(h_1 W^{ff}_1) W^{ff}_2`
   - Feed-forward network
   - Shape: (batch, L, d_model)

10. **h_2** = `h_1 + h_ff`
   - Post-FFN with residual
   - Shape: (batch, L, d_model)

11. **logits** = `h_2 W^{out}`
   - Output logits over vocabulary
   - Shape: (batch, L, vocab_size)

12. **y** = `argmax(logits)`
   - Predicted tokens
   - Shape: (batch, L)
