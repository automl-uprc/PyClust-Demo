# Base image for Python
FROM python

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies in the container
RUN pip install -r requirements.txt

# Copy the entire project into the container (including custom_code)
COPY . .

# Expose the port Gradio will run on
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "tests\gradio_test.py"]
