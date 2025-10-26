# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (if needed for API or dashboard)
EXPOSE 8080

# Run the main script
CMD ["python", "flights_recommender.py"]
