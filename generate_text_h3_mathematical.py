import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'H3')))
import argparse
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from src.models.ssm_seq import SSMLMHeadModel, SSMModel
from src.models.ssm.h3 import H3

# Initialize tokenizer globally so it's accessible in embedding hook
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Let's create a dynamic logger that writes directly to h3_mathematical_trace.md
import json

class MathematicalLogger:
    def __init__(self, filename="h3_mathematical_trace.md", js_filename="h3_index.js"):
        self.filename = filename
        self.js_filename = js_filename
        self.file = open(filename, "w", encoding="utf-8")
        
        # Ensure chunks directory exists
        import os
        os.makedirs("chunks", exist_ok=True)
        self.step_idx = 0
        
        # Open JS index file and start the global index definition
        self.js_file = open(js_filename, "w", encoding="utf-8")
        self.js_file.write("const H3_INDEX_DATA = [\n")
        
        self.write_header()

    def write_header(self):
        self.file.write("# H3 Mathematical Trace Report & Execution Log\n\n")
        self.file.write("> [!NOTE]\n")
        self.file.write("> This report dynamically documents every mathematical equation and the corresponding concrete numerical calculations, shapes, and statistics computed during the execution of the H3 model.\n\n")
        self.file.write("## Mathematical Architecture of H3 (Hungry Hungry Hippos)\n\n")
        self.file.write("The H3 layer processes an input sequence $u(t) \\in \\mathbb{R}^{d}$ using three main steps:\n")
        self.file.write("1. **Linear Projections**: Projects $u(t)$ into Query ($q$), Key ($k$), and Value ($v$) states.\n")
        self.file.write("2. **Shift SSM (K-Kernel)**: Applies a state-space model that acts as a delay/shift register on the Key to capture local context: $k_{shift} = \\text{SSM}_{shift}(k) + D_k \\odot k$.\n")
        self.file.write("3. **Gated Key-Value Multiplication**: Computes the gated interaction: $g(t) = k_{shift}(t) \\odot v(t)$.\n")
        self.file.write("4. **Long-Term Memory SSM (HiPPO/S4D Kernel)**: Resolves long-term history on the gated signal: $y_{ssm} = \\text{SSM}_{S4D}(g) + D \\odot g$.\n")
        self.file.write("5. **Gated Query Multiplication**: Integrates the Query: $y(t) = y_{ssm}(t) \\odot q(t)$.\n")
        self.file.write("6. **Output Projection**: Computes the final layer output: $out(t) = W_o y(t) + b_o$.\n\n")
        self.file.flush()

    def log_section(self, title):
        self.file.write(f"\n---\n\n## {title}\n\n")
        self.file.flush()
        
        # Stream step to JS index
        step = {"type": "section", "title": title}
        self.js_file.write(f"  {json.dumps(step)},\n")
        self.js_file.flush()

    def log_subsection(self, title):
        self.file.write(f"\n### {title}\n\n")
        self.file.flush()
        
        # Stream step to JS index
        step = {"type": "subsection", "title": title}
        self.js_file.write(f"  {json.dumps(step)},\n")
        self.js_file.flush()

    def log_equation(self, latex_eq, description):
        self.file.write(f"**Formula**: {description}\n$$\n{latex_eq}\n$$\n\n")
        self.file.flush()
        
        # Stream step to JS index
        step = {"type": "equation", "latex_eq": latex_eq, "description": description}
        self.js_file.write(f"  {json.dumps(step)},\n")
        self.js_file.flush()

    def log_tensor(self, name, tensor, provenance=None):
        """
        Log a tensor with full provenance metadata.

        provenance: A string describing where this tensor comes from.
          Use one of the following prefixes:
            MODEL      — loaded directly from the checkpoint (learned parameter)
            INPUT      — provided by the user / tokenizer (no computation)
            PREVIOUS_STEP — passed from a prior block, layer, or function
            COMPUTED   — calculated in this step (include formula + inputs)
        """
        shape = list(tensor.shape)
        import math
        if tensor.is_complex():
            mean = "Complex"
            std = "Complex"
        else:
            mean_val = float(tensor.float().mean()) if tensor.is_floating_point() else "N/A"
            if isinstance(mean_val, float):
                if mean_val == float('inf'):
                    mean = "inf"
                elif mean_val == float('-inf'):
                    mean = "-inf"
                elif math.isnan(mean_val):
                    mean = "nan"
                else:
                    mean = mean_val
            else:
                mean = mean_val

            std_val = float(tensor.float().std()) if tensor.is_floating_point() and tensor.numel() > 1 else "N/A"
            if isinstance(std_val, float):
                if std_val == float('inf'):
                    std = "inf"
                elif std_val == float('-inf'):
                    std = "-inf"
                elif math.isnan(std_val):
                    std = "nan"
                else:
                    std = std_val
            else:
                std = std_val
        
        flat = tensor.flatten()
        all_vals = []
        val_strings_joined = ""
        if flat.numel() > 0:
            flat_detach = flat.detach()
            
            # Vectorized float rounding in PyTorch directly on GPU/device for fast execution
            if flat_detach.is_complex():
                flat_cpu = flat_detach.cpu()
                flat_list = flat_cpu.tolist()
                all_vals = [f'"{val.real:.4f}+{val.imag:.4f}j"' if val.imag >= 0 else f'"{val.real:.4f}{val.imag:.4f}j"' for val in flat_list]
                val_strings_joined = ",".join(all_vals)
            elif flat_detach.is_floating_point():
                all_vals = flat_detach.float().cpu().tolist()
                
                # Check for inf / -inf / nan and stringify them correctly with quotes so JSON/JS array is valid!
                formatted_vals = []
                for val in all_vals:
                    if val == float('inf'):
                        formatted_vals.append('"inf"')
                    elif val == float('-inf'):
                        formatted_vals.append('"-inf"')
                    elif math.isnan(val):
                        formatted_vals.append('"nan"')
                    else:
                        formatted_vals.append(str(val))
                val_strings_joined = ",".join(formatted_vals)
            else:
                flat_cpu = flat_detach.cpu()
                all_vals = flat_cpu.tolist()
                val_strings_joined = ",".join(map(str, all_vals))
                
        # Write 100% of all values completely to the Markdown file
        self.file.write(f"- **{name}**:\n")
        self.file.write(f"  - **Shape**: `{shape}`\n")
        self.file.write(f"  - **Mean**: `{mean}` | **StdDev**: `{std}`\n")
        if provenance:
            self.file.write(f"  - **Provenance**: {provenance}\n")
        self.file.write(f"  - **All Values**: `{all_vals}`\n\n")
        self.file.flush()
        
        # Stream step to JS index. Large tensors get stored in a dynamic JSON chunk
        if len(all_vals) > 5000:
            chunk_url = f"chunks/tensor_{self.step_idx}.json"
            with open(chunk_url, "w", encoding="utf-8") as chunk_file:
                chunk_file.write(f"[{val_strings_joined}]")
                
            self.js_file.write("  {\n")
            self.js_file.write('    "type": "tensor",\n')
            self.js_file.write(f'    "name": {json.dumps(name)},\n')
            self.js_file.write(f'    "shape": {json.dumps(shape)},\n')
            self.js_file.write(f'    "mean": {json.dumps(mean)},\n')
            self.js_file.write(f'    "std": {json.dumps(std)},\n')
            if provenance:
                self.js_file.write(f'    "provenance": {json.dumps(provenance)},\n')
            self.js_file.write(f'    "chunk_url": {json.dumps(chunk_url)}\n')
            self.js_file.write("  },\n")
        else:
            # Small tensors embedded directly in the index file for instant loading
            self.js_file.write("  {\n")
            self.js_file.write('    "type": "tensor",\n')
            self.js_file.write(f'    "name": {json.dumps(name)},\n')
            self.js_file.write(f'    "shape": {json.dumps(shape)},\n')
            self.js_file.write(f'    "mean": {json.dumps(mean)},\n')
            self.js_file.write(f'    "std": {json.dumps(std)},\n')
            if provenance:
                self.js_file.write(f'    "provenance": {json.dumps(provenance)},\n')
            self.js_file.write('    "values": [\n')
            self.js_file.write("      " + val_strings_joined + "\n")
            self.js_file.write('    ]\n')
            self.js_file.write("  },\n")
            
        self.js_file.flush()
        self.step_idx += 1

    def log_text(self, text):
        self.file.write(f"{text}\n\n")
        self.file.flush()
        
        # Stream step to JS index
        step = {"type": "text", "content": text}
        self.js_file.write(f"  {json.dumps(step)},\n")
        self.js_file.flush()

    def close(self):
        self.file.close()
        
        # Close JS global array and file
        self.js_file.write("];\n")
        self.js_file.close()

logger = MathematicalLogger()

# Let's intercept and wrap the embedding layer so token embeddings are logged.
original_ssm_forward = SSMModel.forward

def instrumented_ssm_forward(self, input_ids, position_ids=None, inference_params=None):
    hidden_states = self.embeddings(input_ids, position_ids=position_ids)
    logger.log_section("Embedding Layer")
    logger.log_text("Token embeddings are computed by looking up the word embeddings for each token ID in the vocabulary.")
    
    # Log the input ids being processed
    logger.log_tensor(
        "Input Token IDs",
        input_ids,
        provenance=(
            "INPUT: Provided directly by the user's prompt text, tokenized using GPT-2 tokenizer "
            "(tokenizer.encode(prompt)). No model computation involved — these are raw integer token IDs. "
            "Each integer is an index into the vocabulary of size 50257."
        )
    )
    
    # Let's log the individual token lookups
    flat_ids = input_ids.flatten()
    for idx, token_id in enumerate(flat_ids):
        token_id_val = int(token_id.item())
        token_str = tokenizer.decode([token_id_val])
        # Get the corresponding embedding vector from word_embeddings
        word_emb = self.embeddings.word_embeddings.weight[token_id_val]
        
        logger.log_text(f"### Token #{idx + 1}: ID `{token_id_val}` (Text: `\"{token_str}\"`)\\n"
                        f"We retrieve row `{token_id_val}` from the word embedding matrix $W_{{emb}} \\in \\mathbb{{R}}^{{50257 \\times 768}}$:")
        logger.log_tensor(
            f"Word Embedding for \"{token_str}\" (ID {token_id_val})",
            word_emb,
            provenance=(
                f"MODEL: Retrieved directly from the word_embeddings weight matrix "
                f"W_emb ∈ R^(50257 × 768) loaded from checkpoint. "
                f"Formula: emb = W_emb[{token_id_val}, :] — row {token_id_val} of the embedding table. "
                f"Inputs: W_emb (MODEL checkpoint parameter), token_id = {token_id_val} (from Input Token IDs)."
            )
        )
        
    logger.log_text("Since `max_position_embeddings=0`, there are no positional embeddings added. "
                    "Thus, the final Token Embeddings tensor is formed by stacking the individual word embedding vectors.")
    logger.log_tensor(
        "Final Token Embeddings",
        hidden_states,
        provenance=(
            "COMPUTED: Stack of all per-token word embedding vectors for the full input sequence. "
            "Formula: hidden_states = stack([W_emb[t_1], W_emb[t_2], ..., W_emb[t_n]]). "
            "Inputs: Word Embedding for each token (retrieved from W_emb / MODEL). "
            "No positional embeddings are added (max_position_embeddings=0). "
            "Shape: [batch=1, seq_len, d_model=768]. This tensor is passed as hidden_states to the first Block (Layer 0)."
        )
    )
    
    residual = None
    mixer_kwargs = None
    if inference_params is not None:
        mixer_kwargs = dict(inference_params=inference_params)
        
    for layer in self.layers:
        hidden_states, residual = layer(hidden_states, residual, mixer_kwargs=mixer_kwargs)
        
    logger.log_section("Final Block Output & Final Layer Normalization")
    logger.log_text("After passing through all layers, the output of the final layer is added to the running residual stream one last time, followed by the final Layer Normalization:")
    logger.log_equation("residual_{final} = \\text{residual} + \\text{Dropout}(\\text{hidden\\_states}) \\\\ \\text{hidden\\_states}_{final} = \\text{LayerNorm}_{final}(residual_{final})",
                        "Formula for final model residual connection and LayerNorm")
    
    logger.log_tensor(
        "Last Layer hidden_states",
        hidden_states,
        provenance=(
            "PREVIOUS_STEP: 'Block Output hidden_states' returned by the final Block layer (Block N-1). "
            "In that block it is set to mlp_out (if MLP is present) or mixer_out (if MLP is Identity). "
            "Passed here unchanged as the return value of Block.forward()."
        )
    )
    if residual is not None:
        logger.log_tensor(
            "Last Layer residual",
            residual,
            provenance=(
                "PREVIOUS_STEP: 'Block Output residual' returned by the final Block layer (Block N-1). "
                "It equals residual_new2 (= residual_new1 + Dropout(mixer_out)) if MLP is present, "
                "or residual_new1 (= Dropout(hidden_states) + prior_residual) otherwise. "
                "This tensor accumulates the skip-connection highway across every block."
            )
        )
    else:
        logger.log_text("- **Last Layer residual**: `None` (Unexpected state: residual stream is empty)")
        
    if not self.fused_dropout_add_ln:
        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        logger.log_tensor(
            "Final Updated Residual Stream",
            residual,
            provenance=(
                "COMPUTED: residual_final = Dropout(last_layer_hidden_states) + last_layer_residual. "
                "Formula: residual = drop_f(hidden_states) + residual. "
                "Inputs: 'Last Layer hidden_states' (PREVIOUS_STEP — final Block output), "
                "'Last Layer residual' (PREVIOUS_STEP — final Block residual), "
                "drop_f dropout layer (MODEL). "
                "This is the final merged skip-connection before the output LayerNorm."
            )
        )
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        logger.log_tensor(
            "Final Normalized Hidden States",
            hidden_states,
            provenance=(
                "COMPUTED: hidden_states_final = LayerNorm_final(residual_final). "
                "Formula: ln_f(residual.to(ln_f.weight.dtype)). "
                "Inputs: 'Final Updated Residual Stream' (COMPUTED above), "
                "ln_f.weight=Gamma (MODEL checkpoint), ln_f.bias=Beta (MODEL checkpoint), ln_f.eps (MODEL). "
                "This is the model's final output tensor, fed directly to the LM head for next-token prediction."
            )
        )
    else:
        # Import dropout_add_layer_norm in case it's fused
        try:
            from flash_attn.ops.layer_norm import dropout_add_layer_norm
        except ImportError:
            close_match = None
            
        if dropout_add_layer_norm is not None:
            hidden_states = dropout_add_layer_norm(
                hidden_states, residual, self.ln_f.weight, self.ln_f.bias,
                self.drop_f.p if self.training else 0.0, self.ln_f.eps, prenorm=False,
                residual_in_fp32=self.residual_in_fp32
            )
            logger.log_tensor(
                "Final Normalized Hidden States (Fused)",
                hidden_states,
                provenance=(
                    "COMPUTED [Fused Kernel]: Combined dropout + residual add + LayerNorm in a single CUDA kernel "
                    "(flash_attn.ops.layer_norm.dropout_add_layer_norm). "
                    "Equivalent formula: hidden_states = LayerNorm(Dropout(last_hidden) + last_residual). "
                    "Inputs: 'Last Layer hidden_states' (PREVIOUS_STEP), 'Last Layer residual' (PREVIOUS_STEP), "
                    "ln_f.weight=Gamma (MODEL), ln_f.bias=Beta (MODEL), drop_f.p (MODEL), ln_f.eps (MODEL). "
                    "This is the model's final output tensor fed to the LM head."
                )
            )
        else:
            dropped = self.drop_f(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            logger.log_tensor(
                "Final Normalized Hidden States",
                hidden_states,
                provenance=(
                    "COMPUTED: hidden_states_final = LayerNorm_final(Dropout(last_hidden) + last_residual). "
                    "Inputs: 'Last Layer hidden_states' (PREVIOUS_STEP), 'Last Layer residual' (PREVIOUS_STEP), "
                    "ln_f.weight=Gamma (MODEL), ln_f.bias=Beta (MODEL), ln_f.eps (MODEL). "
                    "This is the model's final output tensor fed to the LM head."
                )
            )
            
    return hidden_states

SSMModel.forward = instrumented_ssm_forward

# Let's intercept and wrap the Block forward pass to log how u is calculated from hidden_states and residual!
from flash_attn.modules.block import Block
original_block_forward = Block.forward

def instrumented_block_forward(self, hidden_states, residual=None, mixer_subset=None, mixer_kwargs=None):
    layer_idx = getattr(self, "layer_idx", "Unknown")
    
    logger.log_section(f"Layer {layer_idx} Block: Residual Stream & Input Sequence u Calculation")
    logger.log_text("Before entering the mixer (H3/Attention) layer, the incoming `hidden_states` (output from the previous block or embeddings) is added to the running `residual` stream. This forms the updated residual stream, which is then normalized:")
    logger.log_equation("residual_{new1} = \\text{residual}_{in} + \\text{Dropout}(\\text{hidden\\_states}) \\\\ u = \\text{LayerNorm}_1(residual_{new1})",
                        "Formula for intermediate residual update and input u calculation")
    
    logger.log_tensor(
        "Incoming hidden_states",
        hidden_states,
        provenance=(
            f"PREVIOUS_STEP: For Layer 0, this is 'Final Token Embeddings' from the Embedding Layer. "
            f"For Layer {layer_idx} > 0, this is 'Block Output hidden_states' from Block {layer_idx - 1} "
            f"(either its mlp_out or mixer_out). "
            f"Passed directly as the hidden_states argument to Block.forward(). "
            f"This tensor is NOT yet added to the residual — it feeds into the first skip-connection update."
        )
    )
    if residual is not None:
        logger.log_tensor(
            "Incoming residual stream",
            residual,
            provenance=(
                f"PREVIOUS_STEP: 'Block Output residual' from Block {layer_idx - 1}. "
                f"In Block {layer_idx - 1} it was set to residual_new2 = residual_new1 + Dropout(mixer_out) "
                f"(if MLP exists) or residual_new1 (if no MLP). "
                f"It accumulates the skip-connection highway sum across all preceding blocks. "
                f"Formula that produced it in Block {layer_idx - 1}: "
                f"residual_in[{layer_idx}] = residual_out[{layer_idx - 1}] = residual_new1[{layer_idx - 1}] + Dropout(mixer_out[{layer_idx - 1}])."
            )
        )
        logger.log_text(f"The **Incoming residual stream** for Layer {layer_idx} is the outgoing residual stream of the previous Layer {layer_idx - 1}. "
                        f"In the previous layer (Layer {layer_idx - 1}), it was computed by adding the dropout of the mixer output (or MLP output if present) "
                        f"to the intermediate residual stream:")
        logger.log_equation(f"\\text{{residual}}_{{in, {layer_idx}}} = \\text{{residual}}_{{out, {layer_idx - 1}}} = \\text{{residual}}_{{new1, {layer_idx - 1}}} + \\text{{Dropout}}(\\text{{mixer\\_out}}_{{L{layer_idx - 1}}})",
                            f"Formula showing how the Incoming residual stream for Layer {layer_idx} was created in Layer {layer_idx - 1}")
    else:
        logger.log_text(f"- **Incoming residual stream**: `None` (Initial residual state at Layer 0. Since this is the first block, there is no prior residual stream, so the residual path starts empty and is initialized during the first skip-connection.)")
        
    with torch.no_grad():
        if not self.fused_dropout_add_ln:
            dropped1 = self.drop_path1(self.dropout1(hidden_states))
            residual_new1 = (dropped1 + residual) if residual is not None else dropped1
            logger.log_tensor(
                "Updated Residual Stream (residual + hidden_states)",
                residual_new1,
                provenance=(
                    "COMPUTED: residual_new1 = drop_path1(dropout1(hidden_states)) + residual_in. "
                    "Formula: residual_{new1} = DropPath(Dropout(hidden_states)) + residual_{in}. "
                    "Inputs: 'Incoming hidden_states' (PREVIOUS_STEP — prior block/embedding output), "
                    "'Incoming residual stream' (PREVIOUS_STEP — accumulated skip-connection; None at Layer 0). "
                    "dropout1 and drop_path1 are training-time regularization layers (no-ops during eval). "
                    "This forms the first skip-connection update in this block and is passed to LayerNorm 1."
                )
            )
            
            # LayerNorm 1
            u_calc = self.norm1(residual_new1.to(dtype=self.norm1.weight.dtype))
            logger.log_tensor(
                "Calculated Input Sequence u (LayerNorm 1 Output)",
                u_calc,
                provenance=(
                    "COMPUTED: u = norm1(residual_new1.to(norm1.weight.dtype)). "
                    "Formula: u = γ ⊙ ((residual_new1 − μ) / σ) + β, where μ = per-token mean, σ = per-token std. "
                    "Inputs: 'Updated Residual Stream (residual + hidden_states)' (COMPUTED above), "
                    "'LayerNorm 1 Weight (Gamma)' γ (MODEL), 'LayerNorm 1 Bias (Beta)' β (MODEL), norm1.eps (MODEL). "
                    "This normalized tensor u is fed directly into the H3 mixer (or Attention at attn layers)."
                )
            )
        else:
            dropped1 = self.drop_path1(self.dropout1(hidden_states))
            residual_new1 = (dropped1 + residual) if residual is not None else dropped1
            logger.log_tensor(
                "Updated Residual Stream (residual + hidden_states) [Emulated]",
                residual_new1,
                provenance=(
                    "COMPUTED [Emulated Path — fused_dropout_add_ln=True]: "
                    "Manually computed here only for logging purposes. "
                    "Formula: residual_new1 = drop_path1(dropout1(hidden_states)) + residual_in. "
                    "Inputs: 'Incoming hidden_states' (PREVIOUS_STEP), "
                    "'Incoming residual stream' (PREVIOUS_STEP; None at Layer 0), dropout1, drop_path1. "
                    "In actual model execution this is done inside a fused CUDA kernel — see Mixer Output."
                )
            )
            u_calc = self.norm1(residual_new1.to(dtype=self.norm1.weight.dtype))
            logger.log_tensor(
                "Calculated Input Sequence u (LayerNorm 1 Output) [Emulated]",
                u_calc,
                provenance=(
                    "COMPUTED [Emulated Path — fused_dropout_add_ln=True]: "
                    "u = norm1(residual_new1.to(norm1.weight.dtype)). Same formula as non-fused path. "
                    "Inputs: 'Updated Residual Stream (residual + hidden_states) [Emulated]' (COMPUTED above), "
                    "'LayerNorm 1 Weight (Gamma)' (MODEL), 'LayerNorm 1 Bias (Beta)' (MODEL), norm1.eps (MODEL)."
                )
            )
            
        # Log LayerNorm 1 parameters
        logger.log_tensor(
            "LayerNorm 1 Weight (Gamma)",
            self.norm1.weight,
            provenance=(
                "MODEL: Learned element-wise scale parameter γ of LayerNorm 1, loaded from checkpoint. "
                "Shape [d_model=768]. Applied as γ ⊙ x_normalized inside LayerNorm. "
                "This parameter is NOT computed — it is a trained weight that stays fixed during inference."
            )
        )
        if self.norm1.bias is not None:
            logger.log_tensor(
                "LayerNorm 1 Bias (Beta)",
                self.norm1.bias,
                provenance=(
                    "MODEL: Learned element-wise shift parameter β of LayerNorm 1, loaded from checkpoint. "
                    "Shape [d_model=768]. Applied as + β after γ ⊙ x_normalized inside LayerNorm. "
                    "This parameter is NOT computed — it is a trained weight fixed during inference."
                )
            )
            
        # Calculate true mean/std per-token for ln1
        u_fp32 = residual_new1.float()
        ln1_mean = u_fp32.mean(dim=-1, keepdim=True)
        ln1_var = u_fp32.var(dim=-1, unbiased=False, keepdim=True)
        ln1_std = torch.sqrt(ln1_var + self.norm1.eps)
        logger.log_tensor(
            "LayerNorm 1 Per-Token Mean",
            ln1_mean,
            provenance=(
                "COMPUTED: Per-token mean of residual_new1 over the d_model dimension. "
                "Formula: μ = mean(residual_new1.float(), dim=-1, keepdim=True). "
                "Inputs: 'Updated Residual Stream (residual + hidden_states)' (COMPUTED). "
                "Shape: [batch, seq_len, 1]. Used internally by LayerNorm 1 to center activations. "
                "This is the same μ subtracted inside norm1(residual_new1)."
            )
        )
        logger.log_tensor(
            "LayerNorm 1 Per-Token Std (with epsilon)",
            ln1_std,
            provenance=(
                "COMPUTED: Per-token standard deviation (with epsilon) of residual_new1. "
                "Formula: σ = sqrt(var(residual_new1.float(), dim=-1, unbiased=False) + eps). "
                "Inputs: 'Updated Residual Stream (residual + hidden_states)' (COMPUTED), "
                "norm1.eps (MODEL). Shape: [batch, seq_len, 1]. "
                "Used internally by LayerNorm 1 to scale activations to unit variance."
            )
        )
        logger.log_text(f"- **LayerNorm 1 Epsilon (eps)**: `{self.norm1.eps}`")
        
        # Call mixer (H3 or MHA)
        if mixer_kwargs is None:
            mixer_kwargs = {}
        if mixer_subset is not None:
            mixer_kwargs["mixer_subset"] = mixer_subset
            
        # Execute the mixer (this triggers instrumented_forward / instrumented_step if H3)
        mixer_out = self.mixer(u_calc, **mixer_kwargs)
        if mixer_subset is not None:
            residual_new1 = residual_new1[:, mixer_subset]
            
        logger.log_tensor(
            "Mixer (H3) Output",
            mixer_out,
            provenance=(
                "PREVIOUS_STEP: Output y from the H3 mixer (or Multi-Head Attention at attn_layer_idx layers). "
                "Computed by instrumented_forward() — see the H3 Pre-fill / Convolution Phase section for full derivation. "
                "Final formula: mixer_out = output_linear(y_ssm ⊙ q) = W_o @ (y_ssm ⊙ q) + b_o. "
                "Inputs: 'Calculated Input Sequence u (LayerNorm 1 Output)' (COMPUTED above) → fed as u into H3.forward()."
            )
        )
        
        # Second residual update and MLP
        if not isinstance(self.mlp, nn.Identity):
            logger.log_subsection(f"Layer {layer_idx} Block: MLP & Residual Stream Update")
            logger.log_text("After the mixer layer, the mixer output is added to the residual stream to update it again. This updated residual stream is normalized and passed to the MLP:")
            logger.log_equation("residual_{new2} = residual_{new1} + \\text{Dropout}(\\text{mixer\\_out}) \\\\ x_{mlp} = \\text{LayerNorm}_2(residual_{new2}) \\\\ \\text{hidden\\_states}_{out} = \\text{MLP}(x_{mlp})",
                                "Formula for final block residual update and MLP calculation")
            
            if not self.fused_dropout_add_ln:
                dropped2 = self.drop_path2(self.dropout2(mixer_out))
                residual_new2 = (dropped2 + residual_new1) if residual_new1 is not None else dropped2
                logger.log_tensor(
                    "Updated Residual Stream (residual + mixer_out)",
                    residual_new2,
                    provenance=(
                        "COMPUTED: residual_new2 = drop_path2(dropout2(mixer_out)) + residual_new1. "
                        "Formula: residual_{new2} = DropPath(Dropout(mixer_out)) + residual_{new1}. "
                        "Inputs: 'Mixer (H3) Output' (PREVIOUS_STEP — H3 forward result), "
                        "'Updated Residual Stream (residual + hidden_states)' residual_new1 (COMPUTED earlier). "
                        "dropout2 and drop_path2 are no-ops during eval. "
                        "This is the second skip-connection update in this block. "
                        "It becomes the 'Block Output residual' passed to the next Block as its incoming residual."
                    )
                )
                
                # LayerNorm 2
                ln2_out = self.norm2(residual_new2.to(dtype=self.norm2.weight.dtype))
                logger.log_tensor(
                    "MLP Input (LayerNorm 2 Output)",
                    ln2_out,
                    provenance=(
                        "COMPUTED: ln2_out = norm2(residual_new2.to(norm2.weight.dtype)). "
                        "Formula: x_mlp = γ_2 ⊙ ((residual_new2 − μ_2) / σ_2) + β_2. "
                        "Inputs: 'Updated Residual Stream (residual + mixer_out)' residual_new2 (COMPUTED above), "
                        "'LayerNorm 2 Weight (Gamma)' γ_2 (MODEL), 'LayerNorm 2 Bias (Beta)' β_2 (MODEL), norm2.eps (MODEL). "
                        "This normalized tensor is the direct input to the MLP sublayer."
                    )
                )
            else:
                dropped2 = self.drop_path2(self.dropout2(mixer_out))
                residual_new2 = (dropped2 + residual_new1) if residual_new1 is not None else dropped2
                logger.log_tensor(
                    "Updated Residual Stream (residual + mixer_out) [Emulated]",
                    residual_new2,
                    provenance=(
                        "COMPUTED [Emulated Path — fused_dropout_add_ln=True]: "
                        "Manually computed here only for logging. "
                        "Formula: residual_new2 = drop_path2(dropout2(mixer_out)) + residual_new1. "
                        "Inputs: 'Mixer (H3) Output' (PREVIOUS_STEP), residual_new1 (COMPUTED), dropout2, drop_path2."
                    )
                )
                ln2_out = self.norm2(residual_new2.to(dtype=self.norm2.weight.dtype))
                logger.log_tensor(
                    "MLP Input (LayerNorm 2 Output) [Emulated]",
                    ln2_out,
                    provenance=(
                        "COMPUTED [Emulated Path — fused_dropout_add_ln=True]: "
                        "ln2_out = norm2(residual_new2.to(norm2.weight.dtype)). Same formula as non-fused path. "
                        "Inputs: 'Updated Residual Stream (residual + mixer_out) [Emulated]' (COMPUTED above), "
                        "'LayerNorm 2 Weight (Gamma)' (MODEL), 'LayerNorm 2 Bias (Beta)' (MODEL), norm2.eps (MODEL)."
                    )
                )
                
            # Log LayerNorm 2 parameters
            logger.log_tensor(
                "LayerNorm 2 Weight (Gamma)",
                self.norm2.weight,
                provenance=(
                    "MODEL: Learned element-wise scale parameter γ of LayerNorm 2, loaded from checkpoint. "
                    "Shape [d_model=768]. Applied as γ ⊙ x_normalized inside the second LayerNorm. "
                    "NOT computed — trained weight, fixed during inference."
                )
            )
            if self.norm2.bias is not None:
                logger.log_tensor(
                    "LayerNorm 2 Bias (Beta)",
                    self.norm2.bias,
                    provenance=(
                        "MODEL: Learned element-wise shift parameter β of LayerNorm 2, loaded from checkpoint. "
                        "Shape [d_model=768]. Applied as + β after scaling inside the second LayerNorm. "
                        "NOT computed — trained weight, fixed during inference."
                    )
                )
                
            # Calculate true mean/std per-token for ln2
            ln2_fp32 = residual_new2.float()
            ln2_mean = ln2_fp32.mean(dim=-1, keepdim=True)
            ln2_var = ln2_fp32.var(dim=-1, unbiased=False, keepdim=True)
            ln2_std = torch.sqrt(ln2_var + self.norm2.eps)
            logger.log_tensor(
                "LayerNorm 2 Per-Token Mean",
                ln2_mean,
                provenance=(
                    "COMPUTED: Per-token mean of residual_new2 over the d_model dimension. "
                    "Formula: μ_2 = mean(residual_new2.float(), dim=-1, keepdim=True). "
                    "Inputs: 'Updated Residual Stream (residual + mixer_out)' residual_new2 (COMPUTED). "
                    "Shape: [batch, seq_len, 1]. Used internally by LayerNorm 2 to center activations."
                )
            )
            logger.log_tensor(
                "LayerNorm 2 Per-Token Std (with epsilon)",
                ln2_std,
                provenance=(
                    "COMPUTED: Per-token standard deviation (with epsilon) of residual_new2. "
                    "Formula: σ_2 = sqrt(var(residual_new2.float(), dim=-1, unbiased=False) + eps). "
                    "Inputs: 'Updated Residual Stream (residual + mixer_out)' residual_new2 (COMPUTED), "
                    "norm2.eps (MODEL). Shape: [batch, seq_len, 1]."
                )
            )
            logger.log_text(f"- **LayerNorm 2 Epsilon (eps)**: `{self.norm2.eps}`")
            
            # MLP
            mlp_out = self.mlp(ln2_out)
            logger.log_tensor(
                "MLP Output",
                mlp_out,
                provenance=(
                    "COMPUTED: mlp_out = mlp(ln2_out). "
                    "Formula: MLP(x_mlp) — typically a 2-layer feed-forward: "
                    "mlp_out = W_2 @ GeLU(W_1 @ x_mlp + b_1) + b_2. "
                    "Inputs: 'MLP Input (LayerNorm 2 Output)' ln2_out (COMPUTED above), "
                    "W_1, b_1, W_2, b_2 (MODEL checkpoint — MLP weights and biases). "
                    "Shape: [batch, seq_len, d_model=768]."
                )
            )
            
            out_hidden = mlp_out
            out_residual = residual_new2
        else:
            out_hidden = mixer_out
            out_residual = residual_new1
            
        if self.residual_in_fp32:
            out_residual = out_residual.to(torch.float32)
            
        logger.log_tensor(
            "Block Output hidden_states",
            out_hidden,
            provenance=(
                f"COMPUTED: out_hidden = mlp_out (if MLP is present) OR mixer_out (if MLP is nn.Identity). "
                f"For this block (Layer {layer_idx}), MLP {'is present → out_hidden = MLP Output' if not isinstance(self.mlp, nn.Identity) else 'is Identity → out_hidden = Mixer (H3) Output'}. "
                f"This tensor becomes the 'Incoming hidden_states' for Block {layer_idx + 1}."
            )
        )
        logger.log_tensor(
            "Block Output residual (becomes Incoming residual stream for next block)",
            out_residual,
            provenance=(
                f"COMPUTED: out_residual = residual_new2 (if MLP present) OR residual_new1 (if MLP is Identity). "
                f"For this block (Layer {layer_idx}): "
                f"{'residual_new2 = residual_new1 + Dropout(mixer_out) — see Updated Residual Stream (residual + mixer_out).' if not isinstance(self.mlp, nn.Identity) else 'residual_new1 = Dropout(hidden_states) + residual_in — see Updated Residual Stream (residual + hidden_states).'} "
                f"This becomes the 'Incoming residual stream' for Block {layer_idx + 1}. "
                f"It carries the entire skip-connection highway accumulated so far across all blocks 0 through {layer_idx}."
            )
        )
        
    return out_hidden, out_residual

Block.forward = instrumented_block_forward

# Let's intercept and wrap the H3 forward and step passes!
original_forward = H3.forward
original_step = H3.step

logged_layers = set()
step_count = 0
forward_count = 0

def instrumented_forward(self, u, inference_params=None):
    global forward_count, logged_layers
    forward_count += 1
    logger.log_section(f"Pre-fill / Prompt Convolution Phase (Pass #{forward_count}, Layer {self.layer_idx})")
    logger.log_text("In the pre-fill phase, the input sequence is processed in parallel using Fourier transforms (FFT Convolutions) to build the initial recurrent states and compute prompt outputs.")
    
    logger.log_tensor(
        "Input Sequence u",
        u,
        provenance=(
            f"PREVIOUS_STEP: 'Calculated Input Sequence u (LayerNorm 1 Output)' from the enclosing Block (Layer {self.layer_idx}). "
            f"This is residual_new1 after LayerNorm 1 normalization: u = norm1(residual_new1). "
            f"Passed directly as the u argument to H3.forward(). "
            f"Shape: [batch, seq_len, d_model=768]."
        )
    )
    
    # 1. Linear Projections
    L = u.size(-2)
    u_flat = rearrange(u, 'b l h -> (b l) h')
    dtype = self.q_proj.weight.dtype
    
    logger.log_subsection("1. Projection Step")
    logger.log_equation("q = W_q u + b_q \\\\ k = W_k u + b_k \\\\ v = W_v u + b_v", "Projecting input sequence u to Q, K, and V")
    
    # Log projection weights and biases only once per layer to avoid massive duplication
    if self.layer_idx not in logged_layers:
        logger.log_tensor(
            "Query Projection Weight (W_q)",
            self.q_proj.weight,
            provenance=(
                f"MODEL: Learned weight matrix for the Query linear projection in H3 Layer {self.layer_idx}, "
                f"loaded from checkpoint. Shape [d_model, d_model] = [768, 768]. "
                f"Applied as q = W_q @ u + b_q. NOT computed — trained parameter, fixed during inference."
            )
        )
        if self.q_proj.bias is not None:
            logger.log_tensor(
                "Query Projection Bias (b_q)",
                self.q_proj.bias,
                provenance=(
                    f"MODEL: Learned bias vector for the Query projection in H3 Layer {self.layer_idx}, "
                    f"loaded from checkpoint. Shape [d_model=768]. Added as + b_q after W_q @ u. "
                    f"NOT computed — trained parameter."
                )
            )
        logger.log_tensor(
            "Key Projection Weight (W_k)",
            self.k_proj.weight,
            provenance=(
                f"MODEL: Learned weight matrix for the Key linear projection in H3 Layer {self.layer_idx}, "
                f"loaded from checkpoint. Shape [d_model, d_model] = [768, 768]. "
                f"Applied as k = W_k @ u + b_k. NOT computed — trained parameter."
            )
        )
        if self.k_proj.bias is not None:
            logger.log_tensor(
                "Key Projection Bias (b_k)",
                self.k_proj.bias,
                provenance=(
                    f"MODEL: Learned bias vector for the Key projection in H3 Layer {self.layer_idx}, "
                    f"loaded from checkpoint. Shape [d_model=768]. Added as + b_k after W_k @ u. "
                    f"NOT computed — trained parameter."
                )
            )
        logger.log_tensor(
            "Value Projection Weight (W_v)",
            self.v_proj.weight,
            provenance=(
                f"MODEL: Learned weight matrix for the Value linear projection in H3 Layer {self.layer_idx}, "
                f"loaded from checkpoint. Shape [d_model, d_model] = [768, 768]. "
                f"Applied as v = W_v @ u + b_v. NOT computed — trained parameter."
            )
        )
        if self.v_proj.bias is not None:
            logger.log_tensor(
                "Value Projection Bias (b_v)",
                self.v_proj.bias,
                provenance=(
                    f"MODEL: Learned bias vector for the Value projection in H3 Layer {self.layer_idx}, "
                    f"loaded from checkpoint. Shape [d_model=768]. Added as + b_v after W_v @ u. "
                    f"NOT computed — trained parameter."
                )
            )

    q_out = self.q_proj(u)
    k_out = self.k_proj(u)
    v_out = self.v_proj(u)
    logger.log_tensor(
        "Query (q)",
        q_out,
        provenance=(
            "COMPUTED: q = q_proj(u) = W_q @ u + b_q. "
            "Inputs: 'Input Sequence u' (PREVIOUS_STEP — LayerNorm 1 output), "
            "'Query Projection Weight (W_q)' (MODEL), 'Query Projection Bias (b_q)' (MODEL). "
            "Shape: [batch, seq_len, d_model=768]. "
            "The Query is used at the end of H3 to gate the SSM output: y = y_ssm ⊙ q."
        )
    )
    logger.log_tensor(
        "Key (k)",
        k_out,
        provenance=(
            "COMPUTED: k = k_proj(u) = W_k @ u + b_k. "
            "Inputs: 'Input Sequence u' (PREVIOUS_STEP — LayerNorm 1 output), "
            "'Key Projection Weight (W_k)' (MODEL), 'Key Projection Bias (b_k)' (MODEL). "
            "Shape: [batch, seq_len, d_model=768]. "
            "The Key feeds into the Shift SSM: k_shift = FFTConv(k, K_shift) + D_shift ⊙ k."
        )
    )
    logger.log_tensor(
        "Value (v)",
        v_out,
        provenance=(
            "COMPUTED: v = v_proj(u) = W_v @ u + b_v. "
            "Inputs: 'Input Sequence u' (PREVIOUS_STEP — LayerNorm 1 output), "
            "'Value Projection Weight (W_v)' (MODEL), 'Value Projection Bias (b_v)' (MODEL). "
            "Shape: [batch, seq_len, d_model=768]. "
            "The Value is multiplied with the shifted Key: kv = k_shift ⊙ v (gated key-value product)."
        )
    )

    # -----------------------------------------------------------------------
    # Recompute k_shift intermediate values for detailed logging BEFORE
    # calling original_forward (which consumes k internally).
    # We mirror lines 121-129 of h3.py exactly, under torch.no_grad().
    # -----------------------------------------------------------------------
    logger.log_subsection("2. Shift SSM: k_shift = FFTConv(k, K_shift) + D_shift ⊙ k")
    logger.log_equation(
        "k_{shift} = \\underbrace{\\mathcal{F}^{-1}\\!\\left[\\hat{K}_{shift} \\cdot \\hat{k}\\right]}_{\\text{FFTConv}(k,\\,K_{shift})} + D_{shift} \\odot k",
        "Full decomposition of the Shift SSM output into its two additive components"
    )
    logger.log_text(
        "**How each component is learned / computed during training:**\n\n"
        "| Component | What it is | How it is learned |\n"
        "|-----------|-----------|-------------------|\n"
        "| **K_shift** (Shift SSM kernel) | A short FIR-like kernel computed from the SSM parameters B and C via FFT: `K = iFFT(FFT(B)* · FFT(C))[..., :L]` | B is a **registered learnable parameter** (optimised via gradient descent). C is a learnable `nn.Parameter`. Together they define the delay/shift characteristic of the SSM. |\n"
        "| **FFTConv(k, K_shift)** | Circular convolution of the key sequence k with K_shift, computed efficiently in O(L log L) using the convolution theorem | Not directly learned — it is the result of convolving the projected Key with the learned K_shift kernel. |\n"
        "| **D_shift** (ssm_k_D) | A per-channel scalar skip-connection weight | A standalone `nn.Parameter(torch.randn(d_model))` that is **learned end-to-end** via backprop alongside all other model parameters. |\n"
        "| **D_shift ⊙ k** | Element-wise product of ssm_k_D with the key k; acts as a direct passthrough of k | Not learned — computed from D_shift (learned) and k (computed from W_k). |\n"
        "| **k_shift** | Sum of the two terms above | Emerges from training; the SSM learns to blend the convolved context signal with the instantaneous key via D_shift. |"
    )

    with torch.no_grad():
        # Replicate the shapes used inside H3.forward()
        u_for_kshift = rearrange(u, 'b l h -> (b l) h')
        dtype_kshift = self.q_proj.weight.dtype
        k_raw = self.k_proj.weight @ u_for_kshift.T + self.k_proj.bias.to(dtype_kshift).unsqueeze(-1)
        L_kshift = u.size(-2)
        k_raw = rearrange(k_raw, 'h (b l) -> b h l', l=L_kshift)  # (B, H, L)

        L_kernel_kshift = L_kshift if self.L is None else min(L_kshift, self.L)
        ssm_k_kernel_raw, _ = self.ssm_k_kernel(L=L_kernel_kshift, state=None, rate=1.0)
        ssm_k_kernel_raw = rearrange(ssm_k_kernel_raw, '1 h l -> h l')  # (H, L_kernel)

        logger.log_tensor(
            "K_shift (Shift SSM Convolution Kernel)",
            ssm_k_kernel_raw,
            provenance=(
                f"COMPUTED: ssm_k_kernel = SSKernelShift.forward(L={L_kernel_kshift}). "
                f"Inside SSKernelShift: K = iFFT(FFT(B).conj() * FFT(C))[..., :min(N, L)], zero-padded to L. "
                f"Inputs: B (MODEL — registered OptimModule parameter, shape [H, N=d_state]), "
                f"C (MODEL — nn.Parameter, shape [1, H, N=d_state]). "
                f"Shape: [H=d_model, L_kernel]. "
                f"This is the learned FIR impulse response of the Shift SSM. "
                f"During training: B and C are updated via gradient descent to minimise the task loss — "
                f"they jointly define the kernel K_shift that captures local temporal dependencies in k."
            )
        )

        fft_size_kshift = L_kernel_kshift + L_kshift
        ssm_k_kernel_f = torch.fft.rfft(ssm_k_kernel_raw, n=fft_size_kshift)  # (H, fft_size//2+1) complex
        k_f = torch.fft.rfft(k_raw.to(ssm_k_kernel_raw.dtype), n=fft_size_kshift)  # (B, H, fft_size//2+1)

        logger.log_tensor(
            "FFT of K_shift (frequency-domain kernel K_shift_f)",
            ssm_k_kernel_f,
            provenance=(
                f"COMPUTED: ssm_k_kernel_f = torch.fft.rfft(K_shift, n=fft_size={fft_size_kshift}). "
                f"Inputs: 'K_shift (Shift SSM Convolution Kernel)' (COMPUTED above). "
                f"Shape: [H, {fft_size_kshift}//2+1] complex. "
                f"This is the frequency-domain representation of the Shift SSM kernel. "
                f"Convolution in time domain = element-wise multiplication in frequency domain."
            )
        )
        logger.log_tensor(
            "FFT of k (frequency-domain key k_f)",
            k_f,
            provenance=(
                f"COMPUTED: k_f = torch.fft.rfft(k.to(K_shift.dtype), n=fft_size={fft_size_kshift}). "
                f"Inputs: 'Key (k)' (COMPUTED in Projection Step above), cast to K_shift dtype for numerical consistency. "
                f"Shape: [B, H, {fft_size_kshift}//2+1] complex. "
                f"Represents the key sequence in the frequency domain."
            )
        )

        freq_product = ssm_k_kernel_f * k_f  # element-wise complex multiply (B, H, freq)
        logger.log_tensor(
            "Frequency-Domain Product K_shift_f * k_f",
            freq_product,
            provenance=(
                f"COMPUTED: freq_product = ssm_k_kernel_f * k_f (element-wise complex multiply). "
                f"Inputs: 'FFT of K_shift' (COMPUTED above), 'FFT of k' (COMPUTED above). "
                f"Shape: [B, H, {fft_size_kshift}//2+1] complex. "
                f"By the convolution theorem, multiplying in frequency domain equals convolution in time domain."
            )
        )

        fftconv_out = torch.fft.irfft(freq_product, n=fft_size_kshift)[..., :L_kshift]
        logger.log_tensor(
            "FFTConv(k, K_shift)  [pure convolution output, before D skip]",
            fftconv_out,
            provenance=(
                f"COMPUTED: fftconv_out = torch.fft.irfft(K_shift_f * k_f, n={fft_size_kshift})[..., :{L_kshift}]. "
                f"Inputs: 'Frequency-Domain Product K_shift_f * k_f' (COMPUTED above). "
                f"Shape: [B, H, L={L_kshift}]. "
                f"This is the output of the full circular convolution of k with K_shift, "
                f"representing the contextual, shifted version of the key sequence. "
                f"During training, backprop flows through irfft → rfft → B and C to refine K_shift."
            )
        )

        logger.log_tensor(
            "D_shift (ssm_k_D)  [learned skip-connection scalar per channel]",
            self.ssm_k_D,
            provenance=(
                f"MODEL: ssm_k_D is an nn.Parameter of shape [H=d_model={self.d_model}], "
                f"initialised with torch.randn and loaded from checkpoint. "
                f"One scalar per output channel — controls how much of the raw key k is passed through "
                f"directly (skip connection) alongside the convolved output FFTConv(k, K_shift). "
                f"During training: updated end-to-end by gradient descent via loss → k_shift → kv → SSM → y. "
                f"Formula role: k_shift = FFTConv(k, K_shift) + D_shift ⊙ k, "
                f"where D_shift = ssm_k_D reshaped to [H, 1] for broadcasting over time."
            )
        )

        d_skip_term = rearrange(self.ssm_k_D, 'h -> h 1') * k_raw  # (H,1) * (B,H,L)
        logger.log_tensor(
            "D_shift ⊙ k  [skip-connection term]",
            d_skip_term,
            provenance=(
                f"COMPUTED: d_skip_term = ssm_k_D.reshape(H,1) * k, element-wise (broadcast over batch & time). "
                f"Inputs: 'D_shift (ssm_k_D)' (MODEL — printed above, shape [H={self.d_model}]), "
                f"'Key (k)' (COMPUTED in Projection Step, shape [B, H, L]). "
                f"Shape: [B, H, L={L_kshift}]. "
                f"This skip connection ensures the instantaneous key value is directly added to the SSM output, "
                f"acting like a residual bypass. D_shift is updated by gradient descent during training."
            )
        )

        k_shift_recomputed = fftconv_out + d_skip_term
        logger.log_tensor(
            "k_shift = FFTConv(k, K_shift) + D_shift ⊙ k  [final Shift SSM output]",
            k_shift_recomputed,
            provenance=(
                f"COMPUTED: k_shift = FFTConv(k, K_shift) + D_shift ⊙ k "
                f"= fftconv_out + d_skip_term (element-wise addition). "
                f"Inputs: 'FFTConv(k, K_shift) [pure convolution output]' (COMPUTED above), "
                f"'D_shift ⊙ k [skip-connection term]' (COMPUTED above). "
                f"Shape: [B, H, L={L_kshift}]. "
                f"This is the shifted/contextualised key that feeds into the gated product kv = k_shift ⊙ v. "
                f"During training: The model learns K_shift (via B, C) and D_shift (directly) so that k_shift "
                f"encodes the most useful local temporal context from the key sequence for the downstream SSM."
            )
        )

    # Now call original forward to get the true output y
    y = original_forward(self, u, inference_params=inference_params)
    
    logger.log_subsection("3. Full H3 Forward: kv, S4D SSM, Query Gate, Output Projection")
    logger.log_equation("kv = k_{shift} \\odot v \\\\ y_{ssm} = \\text{FFTConv}(kv, K_{S4D}) + D_{S4D} \\odot kv \\\\ y = \\text{OutputLinear}(y_{ssm} \\odot q)", 
                        "Remaining H3 steps after Shift SSM")
    
    # Log SSM D parameters
    logger.log_tensor(
        "Shift SSM D Parameter (ssm_k_D)",
        self.ssm_k_D,
        provenance=(
            f"MODEL: Learned skip-connection (direct passthrough) coefficient D for the Shift SSM "
            f"(K-kernel) in H3 Layer {self.layer_idx}, loaded from checkpoint. "
            f"Used in formula: k_shift = FFTConv(k, K_shift) + D_shift ⊙ k. "
            f"The D term ensures the input key signal k is directly summed with the convolved output, "
            f"preventing the SSM from losing the instantaneous signal. NOT computed — trained parameter."
        )
    )
    logger.log_tensor(
        "S4D SSM D Parameter (D)",
        self.D,
        provenance=(
            f"MODEL: Learned skip-connection coefficient D for the long-range S4D SSM "
            f"in H3 Layer {self.layer_idx}, loaded from checkpoint. "
            f"Used in formula: y_ssm = FFTConv(kv, K_S4D) + D_S4D ⊙ kv. "
            f"Ensures the gated key-value product kv is directly summed with the SSM convolution output. "
            f"NOT computed — trained parameter."
        )
    )
    
    logger.log_tensor(
        "Output of Layer (y)",
        y,
        provenance=(
            "COMPUTED: Full H3 forward pass output. "
            "Complete computation chain: "
            "k_shift = FFTConv(k, K_shift) + D_shift ⊙ k → "
            "kv = k_shift ⊙ v → "
            "y_ssm = FFTConv(kv, K_S4D) + D_S4D ⊙ kv → "
            "y = output_linear(y_ssm ⊙ q) = W_o @ (y_ssm ⊙ q) + b_o. "
            "Inputs: 'Query (q)', 'Key (k)', 'Value (v)' (all COMPUTED above), "
            "K_shift convolution kernel (MODEL — Shift SSM), "
            "K_S4D convolution kernel (MODEL — S4D SSM), "
            "'Shift SSM D Parameter (ssm_k_D)' (MODEL), "
            "'S4D SSM D Parameter (D)' (MODEL), "
            "'Output Projection Weight (W_o)' (MODEL), "
            "'Output Projection Bias (b_o)' (MODEL). "
            "This value is returned as 'Mixer (H3) Output' to the enclosing Block."
        )
    )
    
    # Log output linear weights and biases only once per layer to avoid duplication
    if self.layer_idx not in logged_layers:
        logger.log_tensor(
            "Output Projection Weight (W_o)",
            self.output_linear.weight,
            provenance=(
                f"MODEL: Learned weight matrix for the H3 output linear projection in Layer {self.layer_idx}, "
                f"loaded from checkpoint. Shape [d_model, d_model] = [768, 768]. "
                f"Applied as W_o @ (y_ssm ⊙ q) to produce the final mixer output. "
                f"NOT computed — trained parameter."
            )
        )
        if self.output_linear.bias is not None:
            logger.log_tensor(
                "Output Projection Bias (b_o)",
                self.output_linear.bias,
                provenance=(
                    f"MODEL: Learned bias vector for the H3 output projection in Layer {self.layer_idx}, "
                    f"loaded from checkpoint. Shape [d_model=768]. "
                    f"Added as + b_o after W_o @ (y_ssm ⊙ q). NOT computed — trained parameter."
                )
            )
        # Mark as fully logged
        logged_layers.add(self.layer_idx)
        
    return y

def instrumented_step(self, u, state_k, state):
    global step_count
    step_count += 1
    # Only trace the first 2 steps to avoid blowing up file size
    if step_count <= 2:
        logger.log_section(f"Recurrent Step Decoding Phase (Token Step #{step_count}, Layer {self.layer_idx})")
        logger.log_text("During generation, the state space model runs in $O(1)$ constant time step-by-step using recurrent state updates.")
        
        logger.log_tensor(
            "Current Input token representation u",
            u,
            provenance=(
                f"PREVIOUS_STEP: Single-token representation at decoding time step {step_count}. "
                f"During autoregressive generation, this is the Block's LayerNorm 1 output (u = norm1(residual_new1)) "
                f"for the current newly generated token, processed through all preceding blocks up to Layer {self.layer_idx}. "
                f"Shape: [batch=1, 1, d_model=768] — only 1 token at a time during recurrent decoding."
            )
        )
        
        # Projections
        q = self.q_proj(u)
        k = self.k_proj(u)
        v = self.v_proj(u)
        logger.log_subsection("1. Recurrent Projection")
        
        # Note: Static projection weights (W_q, W_k, W_v, W_o) are already fully logged in prefill!
        # Here we only log dynamic step activations
        logger.log_tensor(
            "Query q(t)",
            q,
            provenance=(
                f"COMPUTED: q(t) = q_proj(u) = W_q @ u + b_q at decoding step {step_count}. "
                f"Inputs: 'Current Input token representation u' (PREVIOUS_STEP), "
                f"W_q (MODEL — same weights as logged in Pre-fill phase), b_q (MODEL). "
                f"Shape: [batch=1, 1, d_model=768]. Single-token query; used to gate y_ssm(t)."
            )
        )
        logger.log_tensor(
            "Key k(t)",
            k,
            provenance=(
                f"COMPUTED: k(t) = k_proj(u) = W_k @ u + b_k at decoding step {step_count}. "
                f"Inputs: 'Current Input token representation u' (PREVIOUS_STEP), "
                f"W_k (MODEL), b_k (MODEL). Shape: [batch=1, 1, d_model=768]. "
                f"Single-token key; fed into the Shift SSM recurrent update."
            )
        )
        logger.log_tensor(
            "Value v(t)",
            v,
            provenance=(
                f"COMPUTED: v(t) = v_proj(u) = W_v @ u + b_v at decoding step {step_count}. "
                f"Inputs: 'Current Input token representation u' (PREVIOUS_STEP), "
                f"W_v (MODEL), b_v (MODEL). Shape: [batch=1, 1, d_model=768]. "
                f"Single-token value; multiplied with k_shift(t) to form the gated input g(t) = k_shift(t) ⊙ v(t)."
            )
        )
        
        # Execute original step to get outputs
        y, next_state_k, next_state = original_step(self, u, state_k, state)
        
        logger.log_subsection("2. Recurrent SSM State Updates")
        logger.log_equation("x_k(t) = A_k x_k(t-1) + B_k k(t) \\\\ k_{shift}(t) = C_k x_k(t) + D_k k(t) \\\\ g(t) = k_{shift}(t) \\odot v(t) \\\\ x_{S4D}(t) = A x_{S4D}(t-1) + B g(t) \\\\ y_{ssm}(t) = C x_{S4D}(t) + D g(t) \\\\ y(t) = y_{ssm}(t) \\odot q(t)",
                            "Recurrent transition of the state vector and Query-Key-Value gating")
        
        # Log SSM D parameters
        logger.log_tensor(
            "Shift SSM D Parameter (ssm_k_D) [Step]",
            self.ssm_k_D,
            provenance=(
                f"MODEL: Same learned D parameter as logged in the Pre-fill phase for Layer {self.layer_idx}. "
                f"Used in recurrent step formula: k_shift(t) = C_k @ x_k(t) + D_shift ⊙ k(t). "
                f"NOT computed — trained parameter, unchanged between pre-fill and decoding."
            )
        )
        logger.log_tensor(
            "S4D SSM D Parameter (D) [Step]",
            self.D,
            provenance=(
                f"MODEL: Same learned D parameter as logged in the Pre-fill phase for Layer {self.layer_idx}. "
                f"Used in recurrent step formula: y_ssm(t) = C @ x_S4D(t) + D_S4D ⊙ g(t). "
                f"NOT computed — trained parameter, unchanged between pre-fill and decoding."
            )
        )
        
        logger.log_tensor(
            "Recurrent Output y(t)",
            y,
            provenance=(
                f"COMPUTED: Full H3 recurrent step output at decoding step {step_count}. "
                f"Complete computation chain: "
                f"x_k(t) = A_k ⊙ x_k(t-1) + B_k ⊙ k(t) → "
                f"k_shift(t) = C_k @ x_k(t) + D_shift ⊙ k(t) → "
                f"g(t) = k_shift(t) ⊙ v(t) → "
                f"x_S4D(t) = A ⊙ x_S4D(t-1) + B ⊙ g(t) → "
                f"y_ssm(t) = C @ x_S4D(t) + D_S4D ⊙ g(t) → "
                f"y(t) = output_linear(y_ssm(t) ⊙ q(t)). "
                f"Inputs: 'Query q(t)', 'Key k(t)', 'Value v(t)' (all COMPUTED above), "
                f"state_k (PREVIOUS_STEP — Shift SSM state from step {step_count - 1}), "
                f"state (PREVIOUS_STEP — S4D state from step {step_count - 1}), "
                f"A_k, B_k, C_k (MODEL — Shift SSM parameters), "
                f"A, B, C (MODEL — S4D SSM parameters), "
                f"D_shift=ssm_k_D (MODEL), D_S4D=D (MODEL), W_o (MODEL), b_o (MODEL)."
            )
        )
        logger.log_tensor(
            "Updated Shift State state_k",
            next_state_k,
            provenance=(
                f"COMPUTED: New Shift SSM hidden state after processing k(t) at step {step_count}. "
                f"Formula: x_k(t) = A_k ⊙ x_k(t-1) + B_k ⊙ k(t) (diagonal SSM state update). "
                f"Inputs: state_k (PREVIOUS_STEP — Shift SSM state from step {step_count - 1}; "
                f"at step 1 this is initialized from the pre-fill phase), "
                f"'Key k(t)' (COMPUTED above), "
                f"A_k (MODEL — diagonal SSM transition matrix), B_k (MODEL — SSM input matrix). "
                f"This tensor is passed as state_k to the next decoding step (step {step_count + 1})."
            )
        )
        logger.log_tensor(
            "Updated S4D State state",
            next_state,
            provenance=(
                f"COMPUTED: New S4D (long-range) SSM hidden state after processing g(t) at step {step_count}. "
                f"Formula: x_S4D(t) = A ⊙ x_S4D(t-1) + B ⊙ g(t), where g(t) = k_shift(t) ⊙ v(t). "
                f"Inputs: state (PREVIOUS_STEP — S4D state from step {step_count - 1}; "
                f"at step 1 initialized from pre-fill), "
                f"g(t) = k_shift(t) ⊙ v(t) (COMPUTED — gated key-value product), "
                f"A (MODEL — diagonal S4D transition matrix), B (MODEL — S4D input matrix). "
                f"This tensor is passed as state to the next decoding step (step {step_count + 1}). "
                f"It encodes the long-range context accumulated across all prior tokens."
            )
        )
    else:
        y, next_state_k, next_state = original_step(self, u, state_k, state)
        
    return y, next_state_k, next_state

# Apply monkey patching
H3.forward = instrumented_forward
H3.step = instrumented_step

# Main execution logic from generate_text_h3.py
def main():
    parser = argparse.ArgumentParser(description='H3 text generation math logger')
    parser.add_argument('--dmodel', type=int, default=768)
    parser.add_argument('--nlayer', type=int, default=12)
    parser.add_argument('--attn-layer-idx', nargs='+', type=int, default=[6])
    parser.add_argument('--nheads', type=int, default=12)
    parser.add_argument('--ckpt', type=str, default="H3-125M/model.pt")
    parser.add_argument('--genlen', type=int, default=10) # Just 10 tokens to trace
    parser.add_argument('--prompt', type=str, default='Hi H3')
    args = parser.parse_args()

    device = 'cuda'
    dtype = torch.float16

    torch.random.manual_seed(0)
    d_model = args.dmodel
    n_layer = args.nlayer
    ssm_cfg = dict(mode='diag', measure='diag-lin')
    attn_layer_idx = args.attn_layer_idx
    attn_cfg = dict(num_heads=args.nheads)

    logger.log_section("Model Initialization & Checkpoint Loading")
    logger.log_text(f"Initializing SSMLMHeadModel with d_model={d_model}, n_layer={n_layer}, nheads={args.nheads}...")

    model = SSMLMHeadModel(d_model, n_layer=n_layer, d_inner=4 * d_model, vocab_size=len(tokenizer),
                           ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                           pad_vocab_size_multiple=8).to(device=device)

    if args.ckpt is not None and args.ckpt != 'None' and args.ckpt != '':
        state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)
        if 'pytorch-lightning_version' in state_dict:
            state_dict = {k[len('model.'):]: v for k, v in state_dict['state_dict'].items()
                          if k.startswith('model.')}
        
        # Check if checkpoint dimensions match model dimensions
        ckpt_word_embeddings = state_dict.get('backbone.embeddings.word_embeddings.weight')
        if ckpt_word_embeddings is not None and ckpt_word_embeddings.shape[1] != d_model:
            logger.log_text(f"**Warning**: Checkpoint dimension `{ckpt_word_embeddings.shape[1]}` does not match "
                            f"requested d_model `{d_model}`. Skipping checkpoint loading and using randomly "
                            f"initialized weights instead.")
        else:
            model.load_state_dict(state_dict)
            logger.log_text(f"Successfully loaded model weights from checkpoint: `{args.ckpt}`.")
    else:
        logger.log_text("No checkpoint specified. Model initialized with random weights.")

    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.to(dtype=dtype)

    prompt = args.prompt
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device=device)
    max_length = input_ids.shape[1] + args.genlen

    logger.log_section("Generation Start")
    logger.log_tensor(
        "Initial token IDs",
        input_ids,
        provenance=(
            "INPUT: Integer token ID tensor created by tokenizer.encode(prompt), wrapped in unsqueeze(0) "
            "to add a batch dimension. Shape: [1, seq_len]. "
            "No model computation — these are raw vocabulary indices from the GPT-2 tokenizer. "
            "This is the entry point of the entire forward pass."
        )
    )
    logger.log_text(f"Prompt text: `\"{prompt}\"`")

    # Run the generate function which will trigger our instrumented forward/step hooks!
    output_ids = model.generate(input_ids=input_ids, max_length=max_length,
                           return_dict_in_generate=False, output_scores=False, 
                           top_p=0.9, top_k=50, 
                           eos_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.batch_decode(output_ids)[0]
    logger.log_section("Generation Results")
    logger.log_text(f"### Final Generated Text:\n```\n{generated_text}\n```")
    
    logger.close()
    print("\n--------------------------------------------------------------")
    print(f"Mathematical Trace Report successfully written to {logger.filename}!")
    print("--------------------------------------------------------------\n")

from einops import rearrange
if __name__ == '__main__':
    main()
