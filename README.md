# Axolotl GAN - Minimal API

This is a minimal API server for axolotl image generation using a pre-trained GAN model.

## Overview

This project is a stripped-down version of the full axolotl-ai-app, containing only the essential components needed to generate axolotl images using a pre-trained GAN model. It exposes a simple API endpoint for generating images.

## Setup

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure the model files are in the correct location:
   - `data/gan_full_model.pth` (primary model)
   - `data/gan_checkpoint.pth` (fallback model)

## Running the API

### On Windows:
```
.\start.ps1
```

### On Linux/Mac:
```
chmod +x start.sh
./start.sh
```

Or simply:
```
python app.py
```

## API Endpoints

- `/health` - Check if the API is running
- `/generate` - Generate an axolotl image (returns base64 encoded PNG)

## Example Usage

```python
import requests
import base64
from PIL import Image
import io

# Get an image from the API
response = requests.get("http://localhost:5000/generate")
data = response.json()

# Decode the base64 image
img_data = base64.b64decode(data['image'])
img = Image.open(io.BytesIO(img_data))

# Display or save the image
img.show()
img.save("generated_axolotl.png")
```
#   a x o l o t l - m i n i m a l - a p i 
 
 