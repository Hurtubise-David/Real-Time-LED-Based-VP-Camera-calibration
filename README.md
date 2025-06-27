# Real-Time LED-Based Camera Tracking for 3D Virtual Film Production
# Abstract
Virtual Production (VP) using LED volumes and real-time rendering engines has transformed modern filmmaking by enabling the
seamless blending of physical and virtual environments directly on set. However, achieving robust camera calibration in VP requires
more than spatial estimation alone: it also involves managing complex temporal synchronization across multiple subsystems, physical
camera, motion capture tools, and lens metadata sources, each operating on its own internal clock. These components must work
together to estimate the intrinsic and extrinsic parameters of the physical camera with sub-frame precision. This paper presents the
first comprehensive survey of over 100 references, covering not only camera calibration methods but also broader insights into VP
pipelines, real-time rendering, tracking infrastructures, and the integration of emerging technologies for on-set intrinsic and extrinsic
camera calibration. We identify and classify recent methods tailored to VP contexts, spanning geometric solvers, learning-based
pipelines, neural radiance fields (NeRF), moiré-based frequency sensing, and hybrid sensor fusion techniques. Special attention is given
to recent AI-driven advances, including foundation models such as vision transformers and neural priors that estimate camera pose
from monocular input. While these approaches show strong potential, they are rarely evaluated under VP-specific constraints, such
as dynamic lighting, high frame-rate operation, reflective materials, and LED-induced optical interferences. Our analysis highlights
persistent issues with reproducibility, inconsistencies in performance reporting, and a lack of public benchmarking datasets suited to
VP scenarios. To address these limitations, we advocate for the development of LED-aware datasets, cross-modal synchronization
frameworks, and end-to-end calibration-free tracking systems that integrate differentiable rendering and frequency-based sensing.
This survey provides a structured foundation for advancing both academic research and industrial adoption in next-generation VP
systems.

This repository contain the extended tables as shown in the paper: 
David Hurtubise-Martin, Djemel Ziou, and Marie-Flavie Auclair Fortier. 2025. Real-Time LED-Based Camera Tracking for 3D Virtual
Film Production with Foundation Models. 1, 1 (June 2025), 29 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
