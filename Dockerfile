# Use the Python 3.9-slim image
FROM python:3.9-slim

# Set environment variables
ENV PINECONE_API_KEY=pcsk_21QGgS_D3c8eXnTkUjzwdyb2hpH2KkcyP45vL6zPg3KkmVrgJov4uBrgM1rmJ8WisC1CLK

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


RUN mkdir -p /root/.deepface/weights && \
    wget -O /root/.deepface/weights/arcface_weights.h5 \
    https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5 && \
    wget -O /root/.deepface/weights/2.7_80x80_MiniFASNetV2.pth \
    https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth && \
    wget -O /root/.deepface/weights/4_0_0_80x80_MiniFASNetV1SE.pth \
    https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth

# wget -O /root/.deepface/weights/facenet512_weights.h5 \
# https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5 && \


# Copy the rest of your application code into the container
COPY . .

# Expose the Flask app port
EXPOSE 5000

# Define the command to run your Flask application using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "application:app"]


