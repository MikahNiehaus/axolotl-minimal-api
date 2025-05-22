# filepath: c:\prj\axolotl-minimal-api\app.py
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

    def load_model(self, log_list=None):
        """Load the pre-trained generator model from any .pth file in data/"""
        import glob
        # Find any .pth file in the data directory
        model_files = glob.glob(os.path.join(DATA_DIR, '*.pth'))
        if not model_files:
            log("ERROR: No model file found in data/ directory.", log_list)
            raise FileNotFoundError("No model file found in data/ directory.")
        model_path = model_files[0]
        try:
            log(f"Loading model from: {model_path}", log_list)
            checkpoint = torch.load(model_path, map_location=self.device)
            # Try to load only the generator part if possible
            if isinstance(checkpoint, dict) and 'G' in checkpoint:
                self.G.load_state_dict(checkpoint['G'])
                log("Loaded generator weights from 'G' key in checkpoint.", log_list)
            elif isinstance(checkpoint, dict) and 'generator' in checkpoint:
                self.G.load_state_dict(checkpoint['generator'])
                log("Loaded generator weights from 'generator' key in checkpoint.", log_list)
            elif isinstance(checkpoint, dict):
                # Try loading as if the dict is the state_dict
                self.G.load_state_dict(checkpoint)
                log("Loaded generator weights from checkpoint dict directly.", log_list)
            else:
                # Try loading as if the checkpoint is the state_dict
                self.G.load_state_dict(checkpoint)
                log("Loaded generator weights from checkpoint directly.", log_list)
            log("Successfully loaded generator model", log_list)
        except Exception as e:
            log(f"ERROR: Failed to load generator model: {str(e)}", log_list)
            raise RuntimeError(f"Could not load the generator model: {str(e)}")
    
    def generate_image(self, log_list=None):
        """Generate a single image from the model"""
        log("Generating image...", log_list)
        self.G.eval()  # Set to evaluation mode
        
        with torch.no_grad():  # No need to track gradients
            # Generate the image
            fake = self.G(self.fixed_noise).detach().cpu()
            
            # Save to a buffer instead of a file
            buffer = io.BytesIO()
            save_image(fake, buffer, format="PNG", normalize=True)
            buffer.seek(0)
            log("Image generated successfully", log_list)
            
            return buffer.getvalue()

    def generate_grid(self, nrow=3, ncol=3, log_list=None):
        """Generate a single image that is a grid of generated images"""
        log(f"Generating {nrow*ncol} images for grid...", log_list)
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
            log("Grid image generated successfully", log_list)
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
        img_bytes = generator.generate_grid(nrow=3, ncol=3, log_list=logs)
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
