from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles  # Import StaticFiles
from src.pipeline.prediction import Predictor
from src.config.configuration import ConfigurationManager
import os
import base64

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="."), name="static")

def save_temp_image(image_data, file_name):
    # Save the uploaded image data to a temporary file
    with open(file_name, 'wb') as f:
        f.write(image_data)
    return file_name

def get_prediction(image_data):
    # Save the image temporarily
    temp_image_path = "temp_image.jpg"  # Name for temporary image file
    save_temp_image(image_data, temp_image_path)

    config = ConfigurationManager()
    prediction_config = config.get_prediction_config()
    predict = Predictor(prediction_config)

    # Use the file path for prediction
    return predict.make_prediction(temp_image_path)

@app.get("/")
async def main():
    """
    Display the HTML form for image upload.
    """
    content = """
    <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-image: url('/static/background.jpg'); /* Use the static route */
                    background-size: cover; /* Cover the entire page */
                    background-position: center; /* Center the image */
                    background-repeat: no-repeat; /* Prevent repeating the image */
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .container {
                    text-align: center;
                    background: rgba(255, 255, 255, 0.8); /* White background with transparency */
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                h2 {
                    color: #333;
                }
                form {
                    margin: 20px 0;
                }
                input[type="file"] {
                    margin-bottom: 10px;
                }
                input[type="submit"] {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #45a049;
                }
                .result {
                    margin-top: 20px;
                    border: 1px solid #ddd;
                    padding: 15px;
                    background-color: white;
                    border-radius: 5px;
                }
                img {
                    max-width: 500px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Automated Quality Inspection of Casting Products Using Deep Learning</h1>
                <h2>Upload an image for prediction</h2>
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" required>
                    <input type="submit" value="Submit">
                </form>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Handle the image upload and make a prediction.
    """
    try:
        # Save the uploaded file temporarily
        image_data = await file.read()

        # Make the prediction
        predicted_label = get_prediction(image_data)

        # Save the image for display
        temp_image_path = "temp_image.jpg"
        save_temp_image(image_data, temp_image_path)

        # Prepare the HTML response to show the result and the image
        result_content = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-image: url('static/background.jpg'); /* Use the static route */
                        background-size: cover; /* Cover the entire page */
                        background-position: center; /* Center the image */
                        background-repeat: no-repeat; /* Prevent repeating the image */
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                    }}
                    .container {{
                        text-align: center;
                        background: rgba(255, 255, 255, 0.8); /* White background with transparency */
                        padding: 20px;
                        border-radius: 5px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }}
                    h2 {{
                        color: #333;
                    }}
                    .result {{
                        margin-top: 20px;
                        border: 1px solid #ddd;
                        padding: 15px;
                        background-color: white;
                        border-radius: 5px;
                    }}
                    img {{
                        max-width: 500px;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        margin-top: 10px;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 10px;
                        text-decoration: none;
                        color: #4CAF50;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Prediction Result</h2>
                    <p>Predicted Label: <strong>{predicted_label}</strong></p>
                    <h3>Uploaded Image:</h3>
                    <img src="data:image/jpeg;base64,{base64.b64encode(open(temp_image_path, 'rb').read()).decode()}" alt="Uploaded Image">
                    <div>
                    <a href="/">Upload another image</a>

                                        </div>
                </div>
            </body>
        </html>
        """
        # Optionally, remove the temporary file after prediction
        os.remove(temp_image_path)

        return HTMLResponse(content=result_content)
    
    except Exception as e:
        error_content = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-image: url('stastic/background.jpg'); /* Use the static route */
                        background-size: cover; /* Cover the entire page */
                        background-position: center; /* Center the image */
                        background-repeat: no-repeat; /* Prevent repeating the image */
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                    }}
                    .container {{
                        text-align: center;
                        background: rgba(255, 255, 255, 0.8); /* White background with transparency */
                        padding: 20px;
                        border-radius: 5px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }}
                    h2 {{
                        color: #333;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 10px;
                        text-decoration: none;
                        color: #4CAF50;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Error</h2>
                    <p>{str(e)}</p>
                    <a href="/">Go back</a>
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=error_content)
