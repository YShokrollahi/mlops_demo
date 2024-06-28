FROM python:3.8-slim

WORKDIR /app

# Copy the project files into the Docker image
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the Flask app port
EXPOSE 5000

# Run the Flask app
CMD ["python", "scripts/app.py"]
