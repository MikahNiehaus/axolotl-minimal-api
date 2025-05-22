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
        from models.gan_modules import Generator, Discriminator
        self.img_size = 64  # Set to match checkpoint
        self.z_dim = Z_DIM
        self.device = DEVICE
        self.last_seed = None  # Track last used seed
        self.min_seed_diff = int(1e6)  # Minimum difference between seeds
        self.distinctness_scale = 1.0  # Increase to require more difference between grid images
        # Initialize the generator model with correct features_g and img_size
        self.G = Generator(z_dim=Z_DIM, img_channels=3, img_size=64, features_g=64).to(DEVICE)
        self.D = Discriminator(img_channels=3, features_d=64, img_size=64).to(DEVICE)
        self.fixed_noise = self._get_new_noise()
        # Load the model
        self.load_model()

    def _get_new_noise(self):
        import random
        while True:
            new_seed = random.randint(0, 2**31-1)
            if self.last_seed is None or abs(new_seed - self.last_seed) > self.min_seed_diff:
                self.last_seed = new_seed
                break
        torch.manual_seed(self.last_seed)
        return torch.randn(1, self.z_dim, 1, 1, device=self.device)

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
            # Try to load D weights if present (optional, fallback to random D if not found)
            if isinstance(checkpoint, dict) and 'D' in checkpoint:
                try:
                    self.D.load_state_dict(checkpoint['D'])
                    log("Loaded discriminator weights from 'D' key in checkpoint.", log_list)
                except Exception as e:
                    log(f"WARNING: Could not load D weights: {str(e)}", log_list)
            log("Successfully loaded generator model", log_list)
        except Exception as e:
            log(f"ERROR: Failed to load generator model: {str(e)}", log_list)
            raise RuntimeError(f"Could not load the generator model: {str(e)}")

    def generate_image(self):
        """Generate a single image from the model with a seed far from the last"""
        log("Generating image...")
        self.G.eval()  # Set to evaluation mode
        self.fixed_noise = self._get_new_noise()
        with torch.no_grad():  # No need to track gradients
            # Generate the image
            fake = self.G(self.fixed_noise).detach().cpu()
            # Save to a buffer instead of a file
            buffer = io.BytesIO()
            save_image(fake, buffer, format="PNG", normalize=True)
            buffer.seek(0)
            log("Image generated successfully")
            return buffer.getvalue()

    def generate_grid(self, nrow=4, ncol=4):
        """Generate a 4x4 grid of images, ensuring each fools D and is visually distinct from the others, and upscale to 720x720."""
        import torch.nn.functional as F
        from torchvision.transforms.functional import resize
        from torchvision.transforms import InterpolationMode
        log(f"Generating {nrow*ncol} images for grid with D-fooling and distinctness check...")
        self.G.eval()
        self.D.eval()
        images = []
        seeds = []
        max_attempts = 40
        base_threshold = 0.15  # Lower = more strict, adjust as needed
        threshold = base_threshold * self.distinctness_scale
        d_threshold = 0.9  # Only accept images that fool D by a large amount (D output > 0.9)
        for idx in range(nrow * ncol):
            attempt = 0
            while attempt < max_attempts:
                noise = self._get_new_noise()
                with torch.no_grad():
                    fake = self.G(noise)
                    d_out = self.D(fake).view(-1).mean().item()
                    fake_cpu = fake.detach().cpu()
                # Check if it fools D by a large amount
                if d_out <= d_threshold:
                    attempt += 1
                    continue
                # Compare to all previous images
                is_similar = False
                for prev in images:
                    mse = F.mse_loss(fake_cpu, prev).item()
                    if mse < threshold:
                        is_similar = True
                        break
                if not is_similar:
                    images.append(fake_cpu)
                    seeds.append(self.last_seed)
                    break
                attempt += 1
            else:
                # If we can't find a good image, just use the last one
                images.append(fake_cpu)
                seeds.append(self.last_seed)
        images_tensor = torch.cat(images, dim=0)
        from torchvision.utils import make_grid
        grid = make_grid(images_tensor, nrow=nrow, normalize=True, pad_value=1.0, padding=2)
        # Upscale to 720x720 using bilinear interpolation
        grid = resize(grid, [720, 720], interpolation=InterpolationMode.BILINEAR, antialias=True)
        buffer = io.BytesIO()
        save_image(grid, buffer, format="PNG", normalize=True)
        buffer.seek(0)
        log("Grid image generated successfully (4x4, 720x720)")
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
        img_bytes = generator.generate_grid(nrow=4, ncol=4)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        grid_id = f"grid-{int(time.time() * 1000)}"  # Unique grid id
        log("Image generated and encoded successfully", logs)
        return jsonify({
            'image': img_b64,
            'timestamp': str(time.time()),
            'logs': logs,
            'model_type': 'GAN',  # Include the model type for frontend
            'gid': grid_id
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
