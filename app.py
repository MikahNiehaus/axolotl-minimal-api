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

def log(msg, log_list=None):
    """Simple logging function with optional log collection"""
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(log_entry)
    if log_list is not None:
        log_list.append(log_entry)

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

    def generate_grid(self, nrow=3, ncol=3):
        """Generate a single image that is a grid of generated images"""
        log(f"Generating {nrow*ncol} images for grid...")
        self.G.eval()
        with torch.no_grad():
            noise = torch.randn(nrow * ncol, self.z_dim, 1, 1, device=self.device)
            fake = self.G(noise).detach().cpu()
            from torchvision.utils import make_grid
            grid = make_grid(fake, nrow=nrow, normalize=True)
            buffer = io.BytesIO()
            # Save the grid with normalization
            save_image(grid, buffer, format="PNG", normalize=True)
            buffer.seek(0)
            log("Grid image generated successfully")
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
    logs = []
    try:
        log("Received request for image grid generation", logs)
        img_bytes = generator.generate_grid(nrow=3, ncol=3)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        log("Image generated and encoded successfully", logs)
        return jsonify({
            'image': img_b64,
            'timestamp': str(time.time()),
            'logs': logs,
            'model_type': 'GAN'  # Include the model type for frontend
        })
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        log(error_msg, logs)
        return jsonify({
            'error': error_msg,
            'logs': logs,
            'timestamp': str(time.time()),
            'model_type': 'GAN'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    log(f"Starting server on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
