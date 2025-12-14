#!/bin/bash

# Configure Vulkan environment variables
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d

# Configure Unreal Engine's rendering backend to OpenGL
export RHI=OpenGL

# Set to low graphics quality mode (VeryLow / Low / Medium / High / Epic)
export UE4_USE_QUALITY_LEVEL=VeryLow

# Switch to the CARLA directory and start.
cd ~/carla

# Start CARLA and turn off the warning log.
make launch ARGS="-ResX=800 -ResY=600 -windowed -nosound -noloadingscreen -LogCmds='LogCarla warning off'"

