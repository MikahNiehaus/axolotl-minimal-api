from flask import Flask, jsonify
import io
import base64
import os
import torch
from torchvision.utils import save_image
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Constants for the model
IMG_SIZE = 32  # Default image size for the model
Z_DIM = 100    # Noise vector dimension
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    """Simple logging function"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

class MinimalGANGenerator:
    def __init__(self):
        log(f"Initializing MinimalGANGenerator on device: {DEVICE}")
        from models.gan_modules import Generator
        
        self.img_size = IMG_SIZE
        self.z_dim = Z_DIM
        self.device = DEVICE
        
        # Initialize the generator model
        self.G = Generator(z_dim=Z_DIM, img_channels=3, img_size=IMG_SIZE).to(DEVICE)
        self.fixed_noise = torch.randn(1, Z_DIM, 1, 1, device=DEVICE)
        
        # Load the model
        self.load_model()
          def load_model(self):
        """Load the pre-trained model (full model only)"""
        # Path to full model file
        FULL_MODEL_PATH = os.path.join(DATA_DIR, 'gan_full_model.pth')
        
        # Load only the full model - no fallbacks
        if os.path.exists(FULL_MODEL_PATH):
            try:
                log(f"Loading full model from: {FULL_MODEL_PATH}")
                checkpoint = torch.load(FULL_MODEL_PATH, map_location=self.device)
                self.G.load_state_dict(checkpoint['G'])
                log("Successfully loaded full model")
                return
            except Exception as e:
                log(f"ERROR: Failed to load full model: {str(e)}")
                raise RuntimeError(f"Could not load the full model: {str(e)}")
        else:
            log("ERROR: Full model file (gan_full_model.pth) not found.")
            log(f"Expected at path: {FULL_MODEL_PATH}")
            raise FileNotFoundError("Full model file (gan_full_model.pth) not found.")
    
    def generate_image(self):
        """Generate a single image from the model"""
        log("Generating image...")
        self.G.eval()  # Set to evaluation mode
        
        with torch.no_grad():  # No need to track gradients
            # Generate the image
            fake = self.G(self.fixed_noise).detach().cpu()
            
            # Save to a buffer instead of a file
            buffer = io.BytesIO()
            save_image(fake, buffer, format="PNG", normalize=True)
            buffer.seek(0)
            log("Image generated successfully")
            
            return buffer.getvalue()

# Initialize the generator once at startup
generator = MinimalGANGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'timestamp': str(time.time())
    })

@app.route('/generate', methods=['GET'])
def generate_image():
    try:
        log("Received request for image generation")
        img_bytes = generator.generate_image()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return jsonify({
            'image': img_b64,
            'timestamp': str(time.time())
        })
    except Exception as e:
        log(f"Error generating image: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': str(time.time())
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    log(f"Starting server on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
