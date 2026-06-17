TorchServe notes

- Place model artifacts (.mar) into ./torchserve_model_store
- Start with: `docker-compose up torchserve`
- Management API: http://localhost:8081
- Inference API: http://localhost:8080

For GPU-enabled TorchServe builds, build a custom image with CUDA and the matching PyTorch wheel.
