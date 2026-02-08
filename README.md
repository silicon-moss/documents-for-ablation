

### **3.1 The Nemotron-NAS Architecture**

Valkyrie-49B is a fine-tune of nvidia/Llama-3\_3-Nemotron-Super-49B-v1\_5.3 While it is often categorized alongside Llama 3 models due to its tokenizer and general lineage, its internal structure is the product of Neural Architecture Search (NAS). The goal of NAS is to optimize the trade-off between performance and compute/memory efficiency by pruning redundant components from the transformer blocks.4 This results in an irregular, heterogeneous architecture that Heretic cannot navigate.

#### **3.1.1 The Skip-Attention Mechanism**

The most severe incompatibility is the **Skip-Attention** mechanism. In a standard transformer, every block contains a Multi-Head Attention (MHA) module. In Nemotron-NAS, the architecture search algorithm has identified specific layers where the attention mechanism contributes negligibly to the model's predictive accuracy. In these blocks, the attention module is removed entirely or replaced with a simple linear pass-through (identity function).5

This creates a fatal runtime error for Heretic. The tool's modifier.py script typically iterates through the model using a standard loop, expecting to find model.layers\[i\].self\_attn. When it encounters a layer where this attribute is None or a placeholder class without standard weights (e.g., q\_proj, o\_proj), the script crashes with an AttributeError.6 Even if the script were robust enough to handle missing attributes, the mathematical logic of "refusal direction" breaks down. The refusal vector is calculated as the difference in mean activations of the attention outputs. If a layer has no attention output, the vector is undefined or zero. Attempting to project this undefined vector onto subsequent layers (which may have different dimensions) corrupts the residual stream, leading to the "mishaping of 4D tensors" error observed in similar large-scale deployments.1

#### **3.1.2 Variable FFN and Inhomogeneous Layers**

Further compounding the issue is the **Variable FFN** feature. To maximize parameter efficiency, Nemotron-NAS models vary the expansion ratio of the Feed-Forward Networks across different blocks. Layer ![][image6] might have an intermediate dimension of 14,336, while Layer ![][image7] has 10,240.4

Heretic's optimization logic often assumes that the "refusal direction" found in one layer has a geometric relationship to adjacent layers, or it attempts to apply a kernel that smooths weights across dimensions. When the tensor shapes change dynamically between blocks, the matrix operations required for orthogonal projection:

![][image8]  
fail because the dimensions of ![][image9] (derived from Layer ![][image6]) do not match the dimensions of ![][image10] (in Layer ![][image7]). This effectively renders the "MuX" protocol, or any global optimization strategy, mathematically impossible without a rewrite of the tool to handle per-layer dimensionality checks.5

### **3.2 The Infrastructure Collapse: RunPod Storage Dynamics**

While the architectural mismatch guaranteed a software failure, the RunPod environment introduced a layer of operational failure that prevented the user from even reaching the execution phase reliably. The recurring failures in downloading and loading the 100GB model are directly attributable to the mismanagement of RunPod's **Container Disk** vs. **Volume Disk**.

#### **3.2.1 The Ephemeral Container Trap**

RunPod pods are containerized environments (Docker/Kubernetes). They provide two distinct storage areas:

1. **Container Disk:** This holds the OS, libraries, and the /root home directory. It is ephemeral. If the pod is stopped or edited, this disk is wiped and reset to the template image.8  
2. **Volume Disk / Network Volume:** This is persistent storage, typically mounted at /workspace. It survives pod restarts and edits.8

The critical failure point is the default behavior of the Hugging Face transformers library. By default, it caches models in \~/.cache/huggingface, which resides on the **Container Disk**.10 Valkyrie-49B, with 50 billion parameters in BF16, requires approximately **100GB** of disk space for the weights alone (![][image11] bytes).11

Standard RunPod templates often allocate a modest size to the Container Disk (e.g., 20GB or 40GB) to save costs and startup time. When the user initiated the download of Valkyrie, the process filled the root partition (/) before completion. This triggers an OSError: No space left on device and causes the download to terminate or leave corrupted, truncated shard files.12

