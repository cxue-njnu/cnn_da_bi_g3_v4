# CNN_DA_BI_G3_V4 - Model Design (CNN-Encoded Direction-Adaptive Bidirectional
# Connectivity-Conditioned Selective State Model)

Chinese: CNN编码的方向自适应双向脑连接条件化选择性状态模型

Scientific division of labour
1. CNN learns local spectro-temporal patterns inside the 30s STFT.
2. SC / FC provide static spatial and dynamic functional connectivity context on the CNN latent.
3. G3 modulates selective-state dynamics from that brain-network context (retain/update/read).
4. Bidirectional models forward + backward timing inside the whole 30s window.
5. Direction-Adaptive Gate decides how much each clip relies on each direction (not fixed 50:50).
6. Selective Readout truly lets C_t / W_out reconstructions (y_t = W_out(C_t h_t)) drive the final
   representation - not only the final hidden state.

---

## 1. Data and shapes

| tensor | shape | role |
|---|---|---|
| STFT | (B,53,18,70) | train-normalised log-spec |
| FC   | (B,53,18,18) | dynamic functional connectivity per timestep |
| SC   | (18,18)       | static spatial connectivity (shared) |

V2.1 reused + V2.2 incremental features are read as-is; nothing is recomputed.

## 2. Multi-scale STFT CNN (per-electrode, shared)

Input (B,53,18,70) is reshaped to (B*18,1,53,70) - the SAME CNN is applied to every one of 18 EEG
electrodes. NO channel-index convolution.

Multi-scale first layer (three parallel branches, all padding=same, no temporal downsampling):
- Branch A Conv2d(1,32,(3,3))
- Branch B Conv2d(1,32,(3,7))
- Branch C Conv2d(1,32,(5,5))
- each: Conv -> GroupNorm -> GELU; concat -> 96 channels; Conv2d(96,96,1)+GroupNorm+GELU.

Residual local block:
- Depthwise Conv2d(96,96,(3,3), groups=96) -> Pointwise Conv2d(96,96,1) -> GroupNorm -> GELU,
- residual connection; T=53 preserved (no temporal pooling in the CNN).

Frequency compression pools ONLY the frequency axis to one bin (adaptive freq pooling) and project
to give 96-dim H_cnn with time(53) and nodes(18) intact.

Raw STFT skip: H_raw = Linear(70,96)(x); H = LayerNorm(H_cnn + H_raw), H: (B,53,18,96).

Time-reversal equivariance: every local temporal kernel is centre-symmetric over the time axis each
forward (w_sym = 0.5*(w + flip(w,time))). This makes the encoding equivariant to within-window time
reversal, which the bidirectional wrapper relies on (T7).

## 3. SC/FC on CNN latent + Shared Fusion

On the CNN latent:  g_sc[t] = normalize(SC) H[t];  g_fc[t] = normalize(FC[t]) H[t]
giving g_sc, g_fc: (B,53,18,96).

Fusion: Z_t = concat(H_t, g_sc,t, g_fc,t) (288)
-> Linear(288,192) -> LayerNorm -> GELU -> Dropout(0.10) -> Linear(192,128) -> LayerNorm = E_t (B,53,18,128).

## 4. G3 connectivity-conditioned selective SSM

Frozen first version: state_dim=128, context_rank=32, layers=2, inter-layer RMSNorm on,
context_dropout=0.10, lambda_init=0.10.

Base (per layer): u=W_in(E_t), delta_base=W_delta(E_t), B_base=W_B(E_t), C_base=W_C(E_t), A=exp(A_log).

Connectivity conditioning (shared once across all G3 layers):
q_sc=SiLU(W_sc g_sc); q_fc=SiLU(W_fc g_fc);  W_sc,W_fc: 96->32 (no bias);  q=tanh(q_sc+q_fc).
Node-time gate: alpha = sigmoid(W_gate(concat(E_t,q))), alpha:(B,53,18,1).
Global lambda = sigmoid(lambda_logit) shared by all G3 layers.
Bounded residuals: r_delta=tanh(R_delta(q)) etc., R_*:32->128 bias=False.

Final gated selective state:
  delta = softplus(delta_base + lambda*alpha*r_delta).clamp(max=20)
  A=exp(A_log); Abar=exp(-delta*A)
  h_t = Abar*h_{t-1} + delta*B*u;  y_t = W_out(C*h_t)
  B = B_base + lambda*alpha*r_B ;  C = C_base + lambda*alpha*r_C

## 5. Selective Readout (V4 mechanism)

The final classifier is NOT fed just the final hidden state. G3 stores the full
y_seq = W_out(C_t h_t): (B,53,18,128) and h_T: (B,18,128).

Temporal-node selective attention: score_tn = MLP(concat(y_tn,q_tn,alpha_tn)), input 128+32+1,
MLP 161->64->1, softmax over all 53x18 positions: r_seq = sum a_tn y_tn in R^128.
Final hidden/node attention pooling of h_T (not max) -> r_last.
r_dir = LayerNorm(Projection([r_last; r_seq])), Projection 256->128.

## 6. Forward / Backward

CNN encoder + fusion run once -> H, g_sc, g_fc, E.
- Forward G3: time 0..52.
- Backward G3: time 52..0, synchronised reversal of E, g_sc, g_fc, FC-derived context; SC never
  reversed. G3 weights shared.

## 7. Direction-Adaptive Gate + directional heads

Shared scorer psi: s_f=psi([r_f; qsum_f]); s_b=psi([r_b; qsum_b]); [w_f,w_b]=softmax([s_f,s_b]);
r_bi = w_f r_f + w_b r_b. psi shared; qsum = mean summary of (B,53,18,32) brain context.

Direction auxiliary: shared Head_dir:128->1 over r_f / r_b;
L = L_main + gamma*L_dir;  L_dir = 0.5*(BCE(z_f,y)+BCE(z_b,y)); gamma = 0.15.

Direction dropout: train only, per sample P=0.10 forward-only, P=0.10 backward-only, P=0.80 adaptive.
DEV/TEST always both directions + adaptive gate.

Direction LoRA: scaffold, disabled (enable_direction_lora=false). rank 8 on backward W_delta/W_B/W_C.

## 8. Classifier

r_bi -> Linear(128,64) -> GELU -> Dropout(0.20) -> Linear(64,1) = final_logit.
No CNN-direct final fusion; the decision must pass SC/FC -> Bi-G3 -> Direction Gate -> classifier.

## 9. Parameter groups (AdamW, total ~366K parameters)

| group | contents | LR | weight decay | params |
|---|---|---:|---:|---:|
| 1 CNN encoder + fusion | encoder + raw skip + fusion E | 3e-4 | 5e-4 | 119,168 |
| 2 G3 shared/base       | per-layer W_in/W_delta/W_B/W_C/W_out/A_log + norms | 1e-3 | 5e-4 | 165,888 |
| 3 connectivity residual| W_sc/W_fc/W_gate/R_delta/R_B/R_C/lambda_logit | 3e-4 | 1e-3 | 18,594 |
| 4 readout/direction/cls| selective readout, direction scorer, aux head, classifier | 5e-4 | 5e-4 | 62,533 |

Every trainable parameter belongs to exactly one group (coverage == trainable).