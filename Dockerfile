# Use Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /Cat_main

# Copy all files
COPY . /Cat_main

# Install dependencies
RUN pip install streamlit tensorflow pillow numpy tensorflow keras scikit-learn matplotlib opencv-python
RUN apt-get update && apt-get install -y unzip

# Expose port 8501 (Streamlit default)
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# FROM python:3.10-slim

# # Set work directory
# WORKDIR /app

# # Install required Linux packages
# RUN apt-get update && apt-get install -y unzip

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project files
# COPY . .

# # Use environment variables from .env
# # (Only during build, for development use --env-file instead)
# ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
# ENV KAGGLE_KEY=${KAGGLE_KEY}

# # Download dataset (e.g., from Kaggle)
# RUN mkdir -p ~/.kaggle && \
#     echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" > ~/.kaggle/kaggle.json && \
#     chmod 600 ~/.kaggle/kaggle.json && \
#     kaggle datasets download -d patriciabrezeanu/big-cats-image-classification-dataset -p /app/data && \
#     unzip /app/data/*.zip -d /app/data

# # Expose Gradio's default port
# EXPOSE 7860

# # Run Gradio application
# CMD ["python", "app/main.py"]