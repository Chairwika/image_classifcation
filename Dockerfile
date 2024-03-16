# Use the official Tensorflow image as base
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary dependencies
RUN pip install streamlit

# Expose the port streamlit runs on
EXPOSE 8501

# Command to run the streamlit app when the container starts
CMD ["streamlit", "run", "app.py"]
