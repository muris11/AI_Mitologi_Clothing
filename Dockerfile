# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m -u 1000 user

# Copy the rest of the application code
COPY . .

# Set permissions so the user can write to the app directory (IMPORTANT for saving .pkl)
RUN chown -R user:user /app

# Switch to non-root user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 7860

# Start the application
CMD ["python", "server.py"]
