# Use the official Python image from the Docker Hub
FROM python:3.11.10-bullseye

# Install Git
RUN apt-get update && apt-get install -y git && apt-get clean

# Set the working directory in the container
WORKDIR /app
# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]