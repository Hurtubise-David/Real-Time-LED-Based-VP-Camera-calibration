# Real-Time-LED-Based-VP-Camera-calibration
# Abstract
Virtual Production (VP) using LED volumes and real-time rendering engines has revolutionized modern cinematography by enabling
seamless integration of physical and virtual environments. However, accurate and low-latency spatiotemporal camera calibration
estimation remains a core challenge in LED-based VP, due to moiré interferences, rolling shutter distortions, and the absence
of standardized calibration protocols. This paper presents the first comprehensive survey of over 100 methods for intrinsics and
extrinsics parameter calibration tailored to VP contexts, spanning geometric solvers, learning-based pipelines, NeRF models, moiré-
based frequency sensing, and hybrid sensor fusion approaches. Particular attention is given to AI-based developments Foundation
Models (FMs), such as vision transformers, that infer camera parameters from RGB or RGB-D input with minimal supervision. We
examine recent FM-based techniques like FoundationPose which leverage large-scale synthetic datasets, LLM-generated textures, and
contrastive learning to generalize across unseen scenes and domains. While promising, these methods are seldom evaluated under
LED-specific constraints, including optical interferences, lighting variability, and high frame-rate requirements. We identify major
gaps in reproducibility, inconsistent performance metrics, and a lack of LED-VP benchmarking datasets. To address these limitations,
we advocate for the development of standardized datasets and LED-aware training pipelines. We also highlight future directions
combining multimodal FMs, spectral interference analysis, and differentiable rendering for calibration-free, real-time camera tracking.
This survey provides a structured foundation to guide academic research and industry deployment of next-generation multimodal VP
systems powered by FMs.

This repository contain the extended tables as shown in the paper: 
David Hurtubise-Martin, Djemel Ziou, and Marie-Flavie Auclair Fortier. 2025. Real-Time LED-Based Camera Tracking for 3D Virtual
Film Production with Foundation Models. 1, 1 (June 2025), 29 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
