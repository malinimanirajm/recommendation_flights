# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app



# Copy project files
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code, data, and other files
COPY src/ ./src/
COPY tests/ ./tests/


# Set environment variables if needed
ENV PYTHONPATH=/app/src

# Expose port (if needed for API or dashboard)
EXPOSE 8080

# Default command — point to main script inside src/
CMD ["python", "src/flights_recommender.py"]
