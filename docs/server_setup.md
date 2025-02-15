# 🚀 Server Setup Guide

This guide provides a step-by-step script to set up **Python, PyTorch (CUDA 11.8), and JupyterLab** on your server and enable **SSH port forwarding** for remote Jupyter access.

---

## **1️⃣ Update & Upgrade System Packages**
```sh
sudo apt update --fix-missing && sudo apt upgrade -y
```

## **2️⃣ Install Python & PIP**
```sh
sudo apt install python3-pip -y
```

## **3️⃣ Install PyTorch (CUDA 11.8)**
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## **4️⃣ Verify PyTorch GPU Support**
```sh
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

## **5️⃣ Install JupyterLab**
```sh
sudo pip install jupyterlab
```

## **6️⃣ Start JupyterLab**
```sh
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## **7️⃣ SSH Port Forwarding (Access Jupyter Remotely)**
Run this command on your **local machine**:
```sh
ssh -L 8888:localhost:8888 -i ~/.ssh/id_rsa cc@192.5.87.97
```

Then open your browser and go to:
```
http://localhost:8888/lab
```

---

