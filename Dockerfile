# Use an official Python runtime as a parent image
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=src/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV OPENAI_ORGANIZATION=$OPENAI_ORGANIZATION
ENV LANGCHAIN_API_KEY=$LANGCHAIN_API_KEY
ENV LANGCHAIN_PROJECT=$LANGCHAIN_PROJECT

# Run app.py when the container launches
CMD ["flask", "run", "--debug"]
