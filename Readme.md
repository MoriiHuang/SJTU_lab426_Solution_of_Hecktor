
# Hecktor Challenge 2025 Solution - SJTU Lab426


*Official solution for the Hecktor 2025 Medical Image Segmentation Challenge*

## Table of Contents
- [Data Source](#data-source)
- [Quick Start](#quick-start)

## Data Source
Dataset provided by the [PANTHER 2025 Challenge](https://hecktor25.grand-challenge.org/hecktor25/).  
Please register on the challenge website to access the data.

## Quick Start
### Docker Deployment
Download docker image from [Final docker](https://drive.google.com/file/d/1B_BOBolm8is47FPQafmzyfY1nsQcYZkY/view)

```bash
# Run inference
bash run_docker.sh
```

> **Note**: Ensure you have NVIDIA Docker runtime installed for GPU acceleration.

### Pipeline Overview
![Training Pipeline Architecture](images/model.jpg)
