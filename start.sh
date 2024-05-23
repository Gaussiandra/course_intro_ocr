docker run -it \
    -v /media/itsarin/HDD-2TB3/midv500_data:/workspace/midv500_data \
    -v /home/itsarin/Code/a4/ocr/course_intro_ocr:/workspace/course_intro_ocr \
    --name ocr_course \
    --runtime=nvidia \
    --gpus all \
    --device=/dev/nvidia-uvm \
    --device=/dev/nvidia-uvm-tools \
    --device=/dev/nvidia-modeset \
    --device=/dev/nvidiactl \
    --device=/dev/nvidia0 \
    --ipc=host \
    pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
    