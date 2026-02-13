# Models and Libraries Licenses

# Technical & Licensing Review of Selected Computer Vision Models and Libraries  
*Prepared from a senior research / architectural due diligence perspective*

---

# 1. Object Detection Models

## YOLOv8 (Ultralytics)

Developed by: Ultralytics  
Repository: https://github.com/ultralytics/ultralytics  
License: GPL-3.0  

### Technical Overview

YOLOv8 is a modern convolutional neural network (CNN) architecture designed for real-time object detection. It belongs to the YOLO (You Only Look Once) family and is optimized for:

- High detection speed  
- Good accuracy-to-performance ratio  
- Easy deployment  
- GPU acceleration  

Typical use cases:
- License plate detection  
- Object tracking  
- Industrial inspection  
- Surveillance systems  

### Architecture Characteristics

- Anchor-free detection
- Multi-scale feature extraction
- PyTorch-based implementation
- Export support (ONNX, TensorRT, etc.)

### Licensing Considerations (Critical)

YOLOv8 is released under **GPL-3.0**.

Implications:

- If integrated into a distributed product, the entire project must comply with GPL.
- You must release your source code if the product is distributed.
- Not ideal for closed-source commercial software.

### Long-Term Stability

- Actively maintained.
- Technically future-proof.
- Legally restrictive for commercial distribution.

### Recommendation

Use only if:
- Internal tooling
- Research environment
- Open-source project

Avoid if:
- Commercial distributed software
- Proprietary IP product

---

## RT-DETR (Hugging Face, Garon16)

Model page: https://huggingface.co/Garon16/rtdetr_r50vd_russia_plate_detector  
License: Apache-2.0  

### Technical Overview

RT-DETR (Real-Time Detection Transformer) is a transformer-based object detection model derived from the DETR architecture.

Key characteristics:

- Transformer encoder-decoder structure
- No anchor boxes
- End-to-end detection
- Modern architecture aligned with current research trends

Pretrained model example:
- Russian license plate detection

### Licensing

Apache-2.0 license:

- Commercial use allowed
- Modification allowed
- No copyleft requirements
- Safe for proprietary distribution

### Architectural Strength

- Transformer-based (aligned with current SOTA trends)
- Backed by large ecosystem
- Easy integration through Transformers API
- Strong long-term viability

### Risk Assessment

- Individual pretrained model may stop receiving updates
- Core architecture and ecosystem are stable

### Recommendation

Strong candidate for:
- Commercial systems
- Long-term projects
- License-safe deployments

---

# 2. Face Detection Models

## MTCNN (Multi-Task Cascaded CNN)

Repository: https://github.com/ipazc/mtcnn  
License: MIT  

### Technical Overview

MTCNN is a cascaded CNN-based face detector:

- P-Net (proposal)
- R-Net (refinement)
- O-Net (output + landmarks)

Features:
- Face detection
- Landmark extraction
- Lightweight inference

### Strengths

- MIT license (permissive)
- Simple integration
- Good for controlled environments

### Weaknesses

- Older architecture
- Not SOTA by modern standards
- Slower than optimized DNN-based detectors

### Recommendation

Use when:
- Need facial landmarks
- Moderate accuracy acceptable
- Simplicity preferred over cutting-edge accuracy

---

## OpenCV DNN Face Detector (Caffe / ResNet SSD)

License: BSD  

### Technical Overview

Pretrained deep neural network detector using:

- ResNet SSD
- Caffe-based model
- Integrated via OpenCV DNN module

Characteristics:

- No need for heavy ML framework
- CPU-friendly
- Production-stable

### Strengths

- Extremely stable
- Minimal dependencies
- Industrial-grade reliability
- Permissive BSD license

### Weaknesses

- Less flexible for custom training
- Not SOTA accuracy

### Recommendation

Best choice for:
- Long-term enterprise systems
- Low-dependency environments
- Embedded deployments

---

## Haar Cascade Classifier

License: BSD  

### Technical Overview

Classic Viola–Jones detector using:

- Haar-like features
- Integral images
- Boosted cascades

Characteristics:

- Very lightweight
- CPU-only
- Extremely fast

### Limitations

- Poor performance in complex environments
- Sensitive to lighting and angles
- Obsolete for modern applications

### Recommendation

Use only if:
- Hardware extremely constrained
- Accuracy requirements low
- Legacy compatibility needed

---

# 3. OCR

## EasyOCR

Repository: https://github.com/JaidedAI/EasyOCR  
License: Apache-2.0  

### Technical Overview

EasyOCR provides:

- Multilingual OCR
- Deep learning-based recognition
- PyTorch backend
- Simple API

Supported:
- License plates
- Documents
- Scene text

### Strengths

- Apache-2.0 license
- Easy integration
- Active community
- Commercial-friendly

### Weaknesses

- Not fastest for high-throughput enterprise scale
- Limited fine-grained model control

### Recommendation

Ideal for:
- Small to medium-scale OCR
- License plate recognition
- Rapid prototyping

---

# 4. Core Frameworks

## TensorFlow

Website: https://www.tensorflow.org  
License: Apache-2.0  

- Mature
- Production-grade
- Long-term maintained
- Suitable for enterprise

---

## PyTorch

Website: https://pytorch.org  
License: BSD  

- Research-friendly
- Flexible
- Widely adopted
- Commercial-safe

---

## Transformers (Hugging Face)

Documentation: https://huggingface.co/docs/transformers  
License: Apache-2.0  

Provides:
- AutoModelForObjectDetection
- AutoImageProcessor
- Unified model loading API

Strong ecosystem and long-term stability.

---

# 5. Supporting Libraries

## NumPy

Website: https://numpy.org  
License: BSD  

- Core numerical computing library
- Fundamental dependency
- Extremely stable

---

## scikit-learn

Website: https://scikit-learn.org  
License: BSD  

- Evaluation metrics
- Precision / Recall
- Model validation utilities

---

# Final Architectural Assessment

## Safest Long-Term Commercial Stack

Detection:
RT-DETR (Apache-2.0)

Framework:
PyTorch (BSD)

Preprocessing:
OpenCV (BSD)

OCR:
EasyOCR (Apache-2.0)

Evaluation:
NumPy + scikit-learn (BSD)

---

## Risk Summary

Low Risk:
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- TensorFlow
- EasyOCR
- RT-DETR (Apache)

Medium Risk:
- MTCNN (aging architecture)

High Legal Risk:
- YOLOv8 (GPL-3.0)

---

# Executive Conclusion

If building a commercial, long-term supported system:

Recommended stack:
RT-DETR + PyTorch + OpenCV + EasyOCR

If prioritizing maximal long-term stability with minimal licensing risk:
OpenCV DNN + EasyOCR

If internal R&D only:
YOLOv8 acceptable.

---

Prepared with architectural, licensing, and long-term maintenance considerations in mind.
