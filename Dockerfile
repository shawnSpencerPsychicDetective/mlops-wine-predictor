# 1. Start with a base operating system with Python 3.9 installed
# "slim" means a lightweight version to save space
FROM python:3.14-slim

# 2. Set the working directory inside the container
# This is like 'cd /app' inside the box
WORKDIR /app

# 3. Copy dependencies file first
# We do this separately to use Docker caching (makes builds faster)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of our code (app.py and the model folder) into the container
COPY . .

# 6. Define the command to run when the container starts
# host 0.0.0.0 is crucial! It tells the container to listen to outside requests
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]