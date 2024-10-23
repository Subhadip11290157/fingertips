# Use a Python base image (slim for smaller size)
FROM python:3.9-slim

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

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose any required ports (optional, depending on your app)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
