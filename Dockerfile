# Use the Python 3.9-slim image
FROM python:3.9-slim

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set environment variables
ENV PINECONE_API_KEY=pcsk_21QGgS_D3c8eXnTkUjzwdyb2hpH2KkcyP45vL6zPg3KkmVrgJov4uBrgM1rmJ8WisC1CLK
ENV MPLCONFIGDIR="/tmp/matplotlib"  

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Fix broken proxy
RUN echo 'Acquire::http::Pipeline-Depth 0;\nAcquire::http::No-Cache true;\nAcquire::BrokenProxy true;\n' > /etc/apt/apt.conf.d/99fixbadproxy


# Install system dependencies and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 hdf5-tools libhdf5-dev libhdf5-serial-dev wget libxext6 && \
    rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN mkdir -p /tmp/.deepface/.deepface/weights && \
    wget -O /tmp/.deepface/.deepface/weights/arcface_weights.h5 \
    https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5 && \
    wget -O /tmp/.deepface/.deepface/weights/2.7_80x80_MiniFASNetV2.pth \
    https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth && \
    wget -O /tmp/.deepface/.deepface/weights/4_0_0_80x80_MiniFASNetV1SE.pth \
    https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth

# wget -O /root/.deepface/weights/facenet512_weights.h5 \
# https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5 && \


# Copy the rest of your application code into the container
COPY . .

# Change ownership of the application files and DeepFace directory
RUN chown -R appuser:appuser /app && \
    chown -R appuser:appuser /tmp/.deepface && \
    chmod -R 755 /tmp/.deepface && \
    mkdir -p /tmp/matplotlib && \
    chown -R appuser:appuser /tmp/matplotlib

# Switch to non-root user
USER appuser

# Expose the Flask app port
EXPOSE 8787

# # Add healthcheck
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8787/health || exit 1

# Define the command to run your Flask application using Gunicorn
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:8787", "--timeout=120", "src.main:create_face_id_service()"]


