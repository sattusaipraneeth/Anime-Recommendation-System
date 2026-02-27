# Use the official Python image as a base
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the app files into the container
COPY . .

# Install required packages
RUN pip install -r requirements.txt

# Expose the port that Streamlit uses
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 