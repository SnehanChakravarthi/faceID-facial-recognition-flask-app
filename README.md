# Face ID API Server

A robust Face Recognition and Authentication service built with Flask and DeepFace, featuring real-time face detection, anti-spoofing, and secure face matching capabilities.

## Features

- 🔐 Face Authentication & Verification
- 🎯 Real-time Face Detection
- 🛡️ Anti-spoofing Protection
- 🔄 Vector Database Integration (Pinecone)
- 📊 Detailed Processing Metrics
- 🚀 High-performance Face Embedding Generation
- 📝 Comprehensive API Documentation

## Core Technologies

- **Framework**: Flask
- **Face Recognition**: DeepFace
- **Vector Database**: Pinecone
- **Face Recognition Model**: ArcFace
- **Anti-spoofing**: Silent Face Anti-spoofing
- **Container**: Docker

## API Endpoints

### Base URL: `/`

- Method: `GET`
- Description: Service information and API documentation
- Response: Service status, version, and available endpoints

### Health Check: `/health`

- Method: `GET`
- Description: Service health status
- Response: Health status and environment information

### Enrollment: `/api/v1/enroll`

- Method: `POST`
- Content-Type: `multipart/form-data`
- Description: Enroll a new face in the system
- Required Parameters:
  - `image`: Face image file
  - `firstName`: First name
  - `lastName`: Last name
- Optional Parameters:
  - `age`: Age
  - `gender`: Gender
  - `email`: Email address
  - `phone`: Phone number

### Authentication: `/api/v1/authenticate`

- Method: `POST`
- Content-Type: `multipart/form-data`
- Description: Authenticate a face against enrolled faces
- Required Parameters:
  - `image`: Face image file to authenticate

## Quick Start

1. Clone the repository:

2. Set up environment variables:

```bash
export PINECONE_API_KEY=your_pinecone_api_key
```

3. Build and run with Docker:

```bash
docker build -t face-id-api .
docker run -p 8787:8787 -e PINECONE_API_KEY=${PINECONE_API_KEY} face-id-api
```

4. Test the API:

```bash
curl http://localhost:8787/health
```

## Configuration

The server can be configured through various settings in `src/config/settings.py`:

- Face Recognition Models
- Detection Backends
- Matching Thresholds
- Anti-spoofing Settings

## Development Setup

1. Create and activate virtual environment:

Option A - Using Conda: (Recommended)

```bash
# Create new conda environment
conda create -n face-id-api python=3.9
# Activate the environment
conda activate face-id-api
```

Option B - Using venv:

```bash
python3.9 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the development server:

```bash
python src/main.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by Snehan Chakravarthi. For any inquiries or support, feel free to reach out via email at [snehanchakravarthi@gmail.com](mailto:snehanchakravarthi@gmail.com).
