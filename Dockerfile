# Gunakan image python sebagai base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh project file ke work directory
COPY . .

# Expose port yang akan digunakan Flask
EXPOSE 8080

# Jalankan Flask app
CMD ["python", "main.py"]
