# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port (adjust if your app uses a different one)
EXPOSE 5000

# Run the app
CMD ["python", "index.py"]
