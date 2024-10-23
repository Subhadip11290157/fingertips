# Use a Python base image (slim for smaller size)
FROM python:3.10-slim

# Install required system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container to the project root
WORKDIR /Fingertips

# Copy only the essential files and directories
COPY app.py . 
COPY LIP.py .
COPY resources/requirements.txt resources/requirements.txt
COPY resources/notes.txt resources/notes.txt

COPY helpers/ helpers/
COPY header_images/ header_images/
COPY static/ static/
COPY templates/ templates/

# Install the Python dependencies without caching
RUN pip install --no-cache-dir -r resources/requirements.txt

# Cleanup pip cache (if applicable)
RUN rm -rf /root/.cache/pip

# Expose any required ports (optional, depending on your app)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
