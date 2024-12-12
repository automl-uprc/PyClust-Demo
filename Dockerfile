# Base image for Python
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies of the gradio app in the container
# RUN pip install --upgrade pip setuptools wheel
# RUN apt-get update && apt-get install -y gfortran
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container (including custom_code)
COPY . /app/

# Expose the port Gradio will run on
EXPOSE 7860

# Specify the command to run your demo
CMD ["python", "main.py"]


