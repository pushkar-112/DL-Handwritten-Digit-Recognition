# Use full Python image (not slim) to avoid missing libraries
FROM python:3.11

# Install system dependencies for OpenCV and cleanup
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port Flask will run on
EXPOSE 5000

# Command to run the app
CMD ["python", "production_flask_app.py"]
