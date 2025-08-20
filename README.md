# Intrinsic Image Decomposition with VAE + Flow Matching

This repository contains my implementation for **Intrinsic Image Decomposition (IID)** using a **Variational Autoencoder (VAE)** combined with a **Flow Matching Network**.  
The system learns to disentangle input images into **albedo** and **shading**, while training in a lightweight but effective latent space.

---

## 📂 Repository Structure
```
├── data/                          # Dataset (HyperSim processed)
├── docs/
│   └── flow_matching.png           # Illustration of flow matching
├── evaluation_results/
│   ├── evaluation_result_arap.csv
│   ├── evaluation_results_saw.csv
├── logs/                           # Training logs + loss plots
├── results_during_validation/
│   ├── Model_input_output/
│   ├── Shading/
│   └── albedo/
├── requirements.txt
└── README.md
```

---

## 🚀 Setup
Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📖 Method Overview

### 🔹 Variational Autoencoder (VAE)
- Trained on **albedo** images resized to **256×256×3**.  
- Latent space: **12 × 32 × 32**.  
- Loss function combines pixel-wise, perceptual, KL divergence, and adversarial terms.  
- VAE is trained for **41 epochs**.

#### Loss Functions

**Reconstruction (L2):**
$$ \mathcal{L}_{\text{L2}} = \|x - \hat{x}\|_2^2 $$

**Perceptual loss:** (using a fixed feature extractor $\phi$, e.g. VGG)
$$ \mathcal{L}_{\text{perc}} = \|\phi(x) - \phi(\hat{x})\|_2^2 $$

**Kullback–Leibler divergence:**  
with prior $p(z) = \mathcal{N}(0,I)$ and posterior $q_\phi(z \mid x)$
$$ \mathcal{L}_{\text{KL}} = D_{\text{KL}}\big(q_\phi(z\mid x)\;\|\;p(z)\big) $$

**Adversarial loss (generator):**  
(non-saturating GAN formulation for generator)
$$ \mathcal{L}_{\text{GAN}} = -\mathbb{E}_{x}\big[\log D(\hat{x})\big] $$

**Total VAE Objective (with weights):**
$$ \mathcal{L}_{\text{VAE}} = 1\cdot \mathcal{L}_{\text{L2}} + 1\cdot \mathcal{L}_{\text{perc}} + 0.005\cdot \mathcal{L}_{\text{KL}} + 0.1\cdot \mathcal{L}_{\text{GAN}} $$

---

### 🔹 Flow Matching
Flow matching is a method to learn **continuous-time dynamics** that transport samples from a prior (noise) distribution to the target data distribution.  
Unlike diffusion models, which require stochastic sampling and many steps, flow matching directly trains a neural network to predict the **velocity field** of samples along the flow trajectory.

<p align="center">
  <img src="docs/flow_matching.png" width="500"/>
</p>

- U-Net + encoder used for flow prediction.  
- Latent dynamics integrated using **Euler method (10 steps)**.  
- Training loss: **flow matching objective** + **perceptual loss** after decoding.  
- Trained for **200 epochs**.  

---

## 📊 Results

### Validation Samples
- **Input / Output**  
  <p><img src="results_during_validation/Model_input_output/36715.png" width="250"></p>

- **Shading**  
  <p><img src="results_during_validation/Shading/1228160.png" width="250"></p>

- **Albedo**  
  <p><img src="results_during_validation/albedo/425250.png" width="250"></p>

---

### Training Loss Plots
<p align="center">
  <img src="logs/discriminator_loss_plot.png" width="300"/>
  <img src="logs/flow_matching_loss_plot.png" width="300"/>
</p>
<p align="center">
  <img src="logs/train_vae_loss_plot.png" width="300"/>
  <img src="logs/val_vae_loss_plot.png" width="300"/>
</p>

---

### ARAP Dataset Results
<p align="center">
  <img src="docs/arap_results/a_full_comparison.png" width="600"/>
</p>

**Evaluation on ARAP Dataset**

| guidance_scale | method | MSE     | RMSE    | LMSE    | SSIM   |
|----------------|--------|---------|---------|---------|--------|
| 1.0            | euler  | 0.0114  | 0.0945  | 0.0289  | 0.8919 |

---

### SAW Dataset Results

| guidance_scale | method | num_batches | MSE    | RMSE   | LMSE | SSIM  |
|----------------|--------|-------------|--------|--------|------|-------|
| 1.0            | euler  | 1166        | 0.0775 | 0.2522 | inf  | 0.593 |

---

## 📌 Notes
- VAE: ~4M parameters  
- Flow model: ~35M parameters  
- Total inference model: ~37M params (**smaller than typical flow/diffusion models**)

---
