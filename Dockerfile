FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the requirements file into the container
COPY --chown=user requirements.txt .

# Install dependencies specifically using cpu wheels for PyTorch to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860
ENV PORT=7860
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app using python
CMD ["python", "app.py"]
