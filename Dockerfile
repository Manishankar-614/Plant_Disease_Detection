# Use a standard Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy all your project files into the container
# This will copy your app.py, requirements.txt, .h5 model, etc.
COPY . .

# Run pip install in the container
# This will install the full tensorflow, just as you have it
RUN pip install --no-cache-dir -r requirements.txt

# Tell Gunicorn to run your app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]