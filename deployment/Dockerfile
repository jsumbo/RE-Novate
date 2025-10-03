# Use the official Python image as the base
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt file to the working directory
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app and model files into the container
COPY ../streamlit_app /app/streamlit_app
COPY ../model /app/model

# Set Streamlit environment variables for headless mode
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
