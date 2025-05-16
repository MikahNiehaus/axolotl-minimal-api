FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure the data directory exists
RUN mkdir -p data

# Set environment variables
ENV PORT=5000

# Run the application
CMD ["python", "app.py"]
