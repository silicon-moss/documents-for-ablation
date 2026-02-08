# **Root Cause Analysis: Operational Failure of Heretic Workflow on RunPod (Valkyrie-49B) vs. Local Success (Ministral-3B)**

## **1\. Executive Summary and Operational Context**

The systemic failure to operationalize the Heretic abliteration workflow on the Valkyrie-49B model within the RunPod cloud environment, when contrasted with the highly successful local deployment of the Ministral-3B model, represents a definitive case study in the friction between legacy automated safety intervention tools and next-generation, architecture-optimized neural networks. This report provides an exhaustive, root-cause forensic analysis of this divergence, dissecting the failure across three distinct planes: the architectural incompatibility of the Nemotron-NAS lineage with linear probing tools; the ephemeral storage dynamics of containerized cloud environments; and the misalignment of AI-guided remediation strategies which failed to account for the specific topological irregularities of the target model.

The investigation confirms that the "Heretic" tool, while effective on standard dense transformers like Ministral-3B, suffers from a catastrophic failure mode when applied to Neural Architecture Search (NAS) derived models like Valkyrie-49B. This is not merely a resource scaling issue solvable by adding more Video Random Access Memory (VRAM), but a fundamental logic error where the tool’s sequential layer iteration collides with the non-uniform "skip-attention" and "variable Feed-Forward Network (FFN)" blocks characteristic of the Nemotron architecture. Furthermore, the deployment failure was exacerbated by a critical misunderstanding of RunPod’s persistence model, where the default behavior of the Hugging Face cache interacting with ephemeral container disks led to data loss and corruption, indistinguishable to the user from a software crash.

By synthesizing data from local execution logs, cloud infrastructure documentation, and architectural whitepapers, this report establishes a validated execution plan—the "Whispering Red Primate" protocol—to remediate these issues. This necessitates a shift from standard container templates to custom, persistence-aware environments using Network Volumes, and a strategic pivot in model selection or codebase modification to accommodate the irregular tensor geometries of modern efficiency-optimized Large Language Models (LLMs).

## **2\. The Control Case: Deconstructing the Ministral-3B Success**

To accurately diagnose the pathology of the RunPod failure, one must first establish the baseline mechanics of a successful abliteration. The local deployment of the **Ministral-3-3B-Instruct-2512-BF16** model serves as the ideal control variable, isolating the factors of architectural homogeneity and hardware stability that are prerequisites for Heretic’s current operational logic.

### **2.1 Architectural Homogeneity and Tensor Alignment**

The Ministral-3B model, despite its multimodal capabilities involving a vision encoder, fundamentally adheres to the standard dense transformer architecture for its text generation components. This structural uniformity is the implicit assumption upon which the Heretic codebase is constructed.1 In a standard dense model, every decoder layer ![][image1] comprises a self-attention mechanism and a Multi-Layer Perceptron (MLP) block of consistent dimensions. This allows Heretic to iterate sequentially from ![][image2] to ![][image3], calculating the "refusal direction" vector (![][image4]) by comparing activations from harmful and harmless prompts without encountering null references or dimension mismatches.

The success of the local run is evidenced by the seamless execution of the Tree-structured Parzen Estimator (TPE) optimization loop. The logs provided in "New Text Document.txt" reveal a stable search process where the optimizer could sample weights across the model's depth without triggering shape errors.2 For instance, Trial 113 successfully applied an attention output projection (attn.o\_proj) max weight of 1.34 at layer position 16.68 and an MLP down-projection (mlp.down\_proj) max weight of 1.46 at position 17.10. The ability of the tool to target specific "virtual layers" (fractional indices representing interpolation between physical layers) confirms that the underlying tensor stack was uniform, allowing for precise linear algebraic interventions.

### **2.2 The "MuX" Protocol: High-Intensity Ablation**

A critical differentiator in the local success was the application of the "MuX" hyperparameter strategy, a community-derived protocol for aggressive uncensoring. Standard Heretic configurations typically cap the ablation weight (![][image5]) at 2.0 to prevent model degradation. However, the "MuX" findings suggest that dense models require significantly higher intervention intensities to fully excise robust safety alignment.1

The local experiment implemented this by extending the max\_weight search range to 4.5 and employing a strict winsor\_percentage of 0.995. Winsorization is mathematically vital in this context; by clipping the top 0.5% of activation outliers, it prevents the amplified refusal vector from introducing catastrophic noise into the residual stream—a phenomenon colloquially termed "lobotomy".1 The log data validates this approach: Trial 115 achieved a Refusal Rate of 4/100 while maintaining a Kullback-Leibler (KL) divergence of only 0.0266 relative to the base model.2 This low KL score indicates that the model's general reasoning capabilities were preserved almost perfectly, proving that the refusal circuitry in Ministral-3B is topologically distinct and separable via linear projection.

### **2.3 Hardware Stability and Persistence**

The local infrastructure utilized dual NVIDIA RTX 3090 GPUs, providing a combined 48GB of VRAM.1 This setup offered two decisive advantages over the cloud deployment. First, the 48GB capacity provided ample headroom for the 3B model (which requires only \~6GB for weights in BF16), allowing for large batch sizes and the storage of extensive activation histories required for the "Research" module's geometry analysis.

Second, and most critically, the local file system (NTFS/ext4) provided inherent persistence. The model weights were downloaded once to a dedicated path (K:\\ai\\llm-outputs), effectively decoupling the storage lifecycle from the compute lifecycle.2 This meant that if the optimization process crashed or was interrupted, the weights remained intact, eliminating the "download loop" failure mode observed on RunPod. The stability of the local filesystem allowed the 500-trial optimization to run to completion (over 46 minutes) without the risk of storage eviction or network timeout.

## **3\. The Target Failure: Structural Pathology of Valkyrie-49B**

The transition to the Valkyrie-49B model on RunPod represents a crossing of the "complexity gap," where the assumptions of standard tooling no longer align with the reality of state-of-the-art model architectures. Valkyrie-49B is not simply a larger version of Ministral; it is a derivative of the **Nemotron-NAS** lineage, an architecture explicitly designed to violate the homogeneity assumptions of tools like Heretic.

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

#### **3.2.2 The "Edit Pod" Data Loss Cycle**

Users typically respond to "disk full" errors by editing the pod configuration to increase disk size. However, RunPod's architecture dictates that **editing a running pod causes a full reset**, effectively creating a new container instance.8 This action wipes the Container Disk. Consequently, the user increases the disk size, the pod restarts, and the 100GB download they attempted is gone. They start the download again, potentially facing the same issue or running into bandwidth/time constraints. This cycle of ephemeral data loss creates the illusion of a broken workflow when, in reality, it is a configuration error regarding persistence paths.

Furthermore, if the git-lfs (Large File Storage) extension is not correctly installed or initialized in the container, the download may retrieve only the "pointer files" (small text files \~2KB) rather than the actual .safetensors weights.12 Heretic, seeing files present, attempts to load them and crashes with a deserialization error, confusing the user into thinking the model is corrupt or the tool is broken.

### **3.3 The "Gemini" Guidance Failure**

The user query notes a "systemic gemini guidance failure." This refers to the inability of LLM-based coding assistants (like Gemini or ChatGPT) to correctly diagnose this specific stack of issues. The root cause of this guidance failure is **training data lag** and **hallucination of standardization**.

1. **Hallucinated Homogeneity:** AI models are trained heavily on the Llama 2 and Llama 3 architectures, which are standard dense transformers. When asked about "Valkyrie-49B" (a relatively new, niche fine-tune of a specialized NVIDIA model), the AI likely assumed it followed the standard Llama 3 topology. It therefore advised standard Heretic commands, unaware of the Nemotron-NAS skip-attention mechanism that causes the crash.  
2. **Infrastructure Blindness:** Unless explicitly fed the RunPod documentation regarding the distinction between /root (ephemeral) and /workspace (persistent), AI models often default to generic Linux advice. They might suggest pip install or huggingface-cli download without specifying the \--cache-dir /workspace flag, leading the user directly into the Container Disk trap described above.

## **4\. Deep Dive: RunPod Storage Architecture & Persistence**

To formulate a remediation strategy, one must understand the underlying mechanics of RunPod's storage virtualization, which differs significantly from a local hard drive.

### **4.1 Kubernetes Persistence Primitives**

RunPod operates on a Kubernetes-based backend. The "Pod" the user interacts with is a Kubernetes Pod. The **Volume Disk** corresponds to a **Persistent Volume Claim (PVC)** backed by network-attached storage or a local SSD partition on the host node.13

* **Container Filesystem (OverlayFS):** The root filesystem is an overlay. Modifications (like downloading a model to \~/.cache) are written to the "upper" layer of the overlay. When the pod is terminated or restarted (which happens during an "Edit"), this upper layer is discarded, and the pod reverts to the "lower" layer (the base Docker image).15 This is why data on the Container Disk is volatile.  
* **Workspace Mount:** The /workspace directory is a mount point for the persistent volume. Data written here bypasses the overlay filesystem and is committed to the persistent storage medium. This data survives the destruction of the container.9

### **4.2 Network Volumes vs. Local Volume Disks**

RunPod offers two tiers of persistent storage, and the choice impacts workflow stability for large models.

* **Volume Disk (Local):** Tied to the specific pod instance. If the user terminates the pod (to stop billing), the volume is often destroyed unless specific "termination protection" is enabled or the user confuses "Stop" (data preserved) with "Terminate" (data lost).17  
* **Network Volume (Remote):** A crucial solution for large model workflows. This is a decoupled storage entity that exists independently of any pod. It can be mounted to different pods sequentially. This allows a user to download the 100GB Valkyrie model *once* to a Network Volume, terminate the expensive GPU pod, and then re-attach the volume to a cheaper pod later for analysis or a different GPU type for compatibility testing.18

**Table 1: Storage Hierarchy & Failure Risk**

| Feature | Container Disk (Ephemeral) | Volume Disk (Persistent) | Network Volume (Portable) |
| :---- | :---- | :---- | :---- |
| **Path** | /root, /tmp, \~ | /workspace (default) | /workspace (if mounted) |
| **Lifecycle** | Dies with Pod Restart/Edit | Dies with Pod Termination | Independent of Pod |
| **Capacity** | Fixed at Creation | Fixed at Creation | Scalable/Fixed |
| **Risk** | **High:** 100GB download triggers full disk crash. | **Medium:** Data lost if pod terminated. | **Low:** Data persists indefinitely. |
| **Use Case** | OS, installed libs | Project files, active run data | Large datasets (Models) |

## **5\. Architectural Deep Dive: Linear Algebra of the Failure**

The incompatibility between Heretic and Valkyrie can be rigorously defined through the linear algebraic operations involved in the ablation process. This section provides the theoretical justification for why the failure is mathematically inevitable without code modification.

### **5.1 The Orthogonal Projection Assumption**

Heretic removes refusal by projecting the output of a layer weight matrix ![][image10] onto the orthogonal complement of the refusal vector ![][image9]. The operation is:

![][image12]  
This operation implies that ![][image10] exists as a matrix of dimension ![][image13]. In the Nemotron-NAS architecture's skip-attention blocks, the operation for that layer is often reduced to a residual connection:

![][image14]  
where ![][image15] might be ![][image16] (identity skip) or a simplified linear transform to match dimensions.5 If Heretic attempts to locate ![][image17] (the output projection of the attention head) in a layer where attention has been pruned, it attempts to perform matrix multiplication on a non-existent object.

### **5.2 Dimensional Invariance Violation**

Furthermore, the calculation of the refusal vector ![][image9] depends on the mean difference of activations:

![][image18]  
In a "Variable FFN" architecture, the dimension of the hidden states ![][image19] inside the MLP block changes dynamically. Let ![][image20].

* In Llama-3 (Ministral), ![][image21] is constant across all layers.  
* In Nemotron (Valkyrie), ![][image22].

Standard ablation tools often aggregate gradients or activations across layers to find a "global" refusal direction to improve stability (smoothing). If the tool attempts to transfer a direction found in layer ![][image23] to layer ![][image24], or compute a global average, the vector dimensions will not align:

![][image25]  
Since ![][image22], the operation is undefined in vector space, causing the "shape mismatch" or "4D tensor" errors observed in the logs of users attempting to ablate similar NAS-based models.1

## **6\. Execution Plan: The "Whispering Red Primate" Protocol**

To successfully remediate these failures and operationalize a Heretic-like workflow on RunPod for large, irregular models, we must implement a rigorous "DevOps" approach. This protocol, designated "Whispering Red Primate," addresses both the infrastructure persistence issues and the architectural incompatibilities.

### **6.1 Phase 1: Infrastructure & Persistence Configuration**

The first step is to eliminate the variable of data loss. We must decouple the model storage from the compute instance.

1. **Provision Network Volume:**  
   * Navigate to RunPod Storage. Create a **Network Volume** (not just a volume disk) of **200GB**.  
   * Name: persistent-llm-storage.  
   * Region: Must match the region of the intended GPU pod (e.g., US-NJ).  
2. **Deploy Template with Environment Overrides:**  
   * Do not use a generic template. Create a **Custom Template** or use the "RunPod PyTorch" template but modify the **Environment Variables** at launch.  
   * **Crucial Step:** Set the Hugging Face cache directory to the persistent volume.  
   * Key: HF\_HOME  
   * Value: /workspace/huggingface\_cache  
   * Key: TORCH\_HOME  
   * Value: /workspace/torch\_cache

This ensures that when transformers downloads the 100GB Valkyrie model, it writes to the persistent Network Volume mounted at /workspace, completely bypassing the ephemeral container disk.10

3. **Hardware Provisioning:**  
   * **Requirement:** 2x NVIDIA A100 (80GB) or 4x A6000.  
   * **Why:** Valkyrie-49B (BF16) is \~100GB. A single A100 (80GB) will crash with OOM immediately. 2x A100 provides 160GB, sufficient for the model \+ activations \+ Heretic's overhead.20

### **6.2 Phase 2: Software Remediation (Patching Heretic)**

Since Heretic v1.x does not natively support Nemotron-NAS, we must apply a "field patch" to the code or switch targets.

**Option A: The "Shim" Patch (For Advanced Users)**

The user must fork Heretic and modify modifier.py to handle NoneType attention layers.

* *Logic:* Wrap the layer iteration in a try/except AttributeError block.  
* *Code:*  
  Python  
  \# Conceptual patch for modifier.py  
  for i, layer in enumerate(model.layers):  
      if not hasattr(layer, 'self\_attn') or layer.self\_attn is None:  
          print(f"Skipping ablation on layer {i}: No attention mechanism.")  
          continue  
      \#... proceed with ablation

This allows the script to skip the optimized NAS layers without crashing, applying ablation only to the standard blocks. This is a heuristic fix but allows the workflow to proceed.

**Option B: The Strategic Pivot (Recommended)**

If the specific architectural features of Nemotron are not required, pivot the target model to **Llama-3.1-70B-Instruct**.

* **Rationale:** It fits on the same 2x A100 hardware profile. It uses a standard, homogeneous dense architecture that is 100% compatible with Heretic. It offers superior reasoning performance to the 49B model. This eliminates the architectural failure mode entirely while maintaining the scale of the experiment.

### **6.3 Phase 3: The "MuX" Parameter Scaling**

Once the infrastructure is stable and the architecture is compatible (or patched), apply the "MuX" parameters verified in the Ministral run, scaled for the larger model.

* **Max Weight:** 4.0 \- 6.0 (Larger models often tolerate higher absolute weights due to deeper residual streams).  
* **Winsorization:** 0.995 (Non-negotiable for stability).  
* **Layers:** Focus on the middle-to-late layers (e.g., layers 20-60 on a 70B model), as early layers encode syntax and late layers encode output formatting, while refusal logic typically resides in the upper-middle depth.

## **7\. Strategic Implications and Gemini Guidance**

The failure of the "Gemini guided" workflow highlights a critical gap in current AI-assisted DevOps. The AI failed to distinguish between "RunPod the container" and "RunPod the workspace," leading to data loss. It also failed to distinguish between "Llama 3 the brand" and "Nemotron the architecture," leading to software crashes.

**Implication:** For future workflows, "Gemini instructions" must be explicitly grounded in the specific hardware documentation (e.g., referencing RunPod's storage classes) and the specific model config (e.g., checking config.json for architectures: \["NemotronForCausalLM"\] vs \["LlamaForCausalLM"\]) before generating code. Blind reliance on high-level abstractions ("It's a Llama model on a Cloud GPU") is the root cause of these systemic failures.

## **8\. Conclusion**

The root cause of the Heretic workflow failure on RunPod with Valkyrie-49B is a compound fracture of **Architecture**, **Infrastructure**, and **Assumption**.

1. **Architecture:** The **Nemotron-NAS** topology (Skip-Attention, Variable FFN) creates a jagged tensor landscape that Heretic’s linear, sequential logic cannot traverse, causing crashes that do not occur on the uniform Ministral-3B.  
2. **Infrastructure:** The 100GB weight of the model crushed the ephemeral **Container Disk** limits of the RunPod instance, while the lack of **Network Volume** configuration meant that every recovery attempt resulted in total data loss.  
3. **Assumption:** The user (and their AI guide) treated a hyper-optimized, irregular 49B model as if it were a standard dense 3B model, failing to provision the **2x A100** hardware required for basic inference, let alone the memory-intensive ablation process.

**Final Recommendation:** To achieve the goal of a highly capable, abliterated model on RunPod, the user should execute the **"Whispering Red Primate"** protocol: Provision a **2x A100 Pod** with a **200GB Network Volume**, set HF\_HOME to /workspace, and pivot the target model to **Llama-3.1-70B** to ensure architectural compliance with the Heretic toolset. This aligns the hardware capacity, storage persistence, and software logic with the physical reality of the workload.

#### **Works cited**

1. Heretic Local Model Experimentation Plan  
2. New Text Document.txt  
3. TheDrummer/Valkyrie-49B-v2 \- Hugging Face, accessed February 7, 2026, [https://huggingface.co/TheDrummer/Valkyrie-49B-v2](https://huggingface.co/TheDrummer/Valkyrie-49B-v2)  
4. nvidia/Llama-3\_1-Nemotron-51B-Instruct · Hugging Face, accessed February 7, 2026, [https://huggingface.co/nvidia/Llama-3\_1-Nemotron-51B-Instruct](https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct)  
5. nvidia/Llama-3\_3-Nemotron-Super-49B-v1 \- Hugging Face, accessed February 7, 2026, [https://huggingface.co/nvidia/Llama-3\_3-Nemotron-Super-49B-v1](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1)  
6. p-e-w/heretic: Fully automatic censorship removal for ... \- GitHub, accessed February 7, 2026, [https://github.com/p-e-w/heretic](https://github.com/p-e-w/heretic)  
7. nvidia/Llama-3\_1-Nemotron-Ultra-253B-v1 \- Hugging Face, accessed February 7, 2026, [https://huggingface.co/nvidia/Llama-3\_1-Nemotron-Ultra-253B-v1](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1)  
8. Storage options \- Runpod Documentation, accessed February 7, 2026, [https://docs.runpod.io/pods/storage/types](https://docs.runpod.io/pods/storage/types)  
9. Overview \- Runpod Documentation, accessed February 7, 2026, [https://docs.runpod.io/pods/overview](https://docs.runpod.io/pods/overview)  
10. How to cache model download from HuggingFace \- Tips? \- Runpod \- Answer Overflow, accessed February 7, 2026, [https://www.answeroverflow.com/m/1311801267839828030](https://www.answeroverflow.com/m/1311801267839828030)  
11. TheDrummer/Valkyrie-49B-v2.1-GGUF \- Hugging Face, accessed February 7, 2026, [https://huggingface.co/TheDrummer/Valkyrie-49B-v2.1-GGUF](https://huggingface.co/TheDrummer/Valkyrie-49B-v2.1-GGUF)  
12. I cannot seem to run any workflow on runpod on comfyui \- Models \- Hugging Face Forums, accessed February 7, 2026, [https://discuss.huggingface.co/t/i-cannot-seem-to-run-any-workflow-on-runpod-on-comfyui/172239](https://discuss.huggingface.co/t/i-cannot-seem-to-run-any-workflow-on-runpod-on-comfyui/172239)  
13. StatefulSets \- Kubernetes, accessed February 7, 2026, [https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)  
14. Runpod vs. Hyperstack: Which Cloud GPU Platform Is Better for Fine-Tuning AI Models?, accessed February 7, 2026, [https://www.runpod.io/articles/comparison/runpod-vs-hyperstack-fine-tuning](https://www.runpod.io/articles/comparison/runpod-vs-hyperstack-fine-tuning)  
15. Chapter 2\. Understanding ephemeral storage | Storage | OpenShift Container Platform | 4.8, accessed February 7, 2026, [https://docs.redhat.com/en/documentation/openshift\_container\_platform/4.8/html/storage/understanding-ephemeral-storage](https://docs.redhat.com/en/documentation/openshift_container_platform/4.8/html/storage/understanding-ephemeral-storage)  
16. Persist data outside of containers \- Runpod Documentation, accessed February 7, 2026, [https://docs.runpod.io/tutorials/introduction/containers/persist-data](https://docs.runpod.io/tutorials/introduction/containers/persist-data)  
17. Manage Pods \- Runpod Documentation, accessed February 7, 2026, [https://docs.runpod.io/pods/manage-pods](https://docs.runpod.io/pods/manage-pods)  
18. Network volumes \- Runpod Documentation, accessed February 7, 2026, [https://docs.runpod.io/storage/network-volumes](https://docs.runpod.io/storage/network-volumes)  
19. Optimizing Docker Setup for PyTorch Training with CUDA 12.8 and Python 3.11 \- Runpod, accessed February 7, 2026, [https://www.runpod.io/articles/guides/docker-setup-pytorch-cuda-12-8-python-3-11](https://www.runpod.io/articles/guides/docker-setup-pytorch-cuda-12-8-python-3-11)  
20. Run Very Large LLMs Securely with RunPod Serverless, accessed February 7, 2026, [https://www.runpod.io/blog/runpod-serverless-secure-llms](https://www.runpod.io/blog/runpod-serverless-secure-llms)
