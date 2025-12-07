# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
# ENV GEMINI_API_KEY="your_api_key_here" 
# (Better to pass this at runtime via --env-file or -e)

# Run server.py when the container launches
CMD ["python", "server.py"]
