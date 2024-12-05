# Base image for Python
FROM python

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies in the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container (including custom_code)
COPY . /app/

RUN pip install -e ./local_library

# Expose the port Gradio will run on
EXPOSE 7860

# Specify the command to run your demo
CMD ["python", "pyclust_demo/main.py"]


