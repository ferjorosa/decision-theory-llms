# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the APT requirements file from the build context (same directory as Dockerfile)
COPY apt.txt .
RUN apt-get update && \
    xargs -a apt.txt apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file from the build context
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Gradio application script from the build context
COPY gradio_app.py .

# Make port 7860 available (Gradio's default port)
EXPOSE 7860

# Run your Gradio app when the container launches, with unbuffered output
CMD ["python", "-u", "gradio_app.py"]