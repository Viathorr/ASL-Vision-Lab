from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from api.inference import predict

app = FastAPI()

@app.post("/predict/")
async def get_prediction(file: UploadFile = File(...)):
  """
  Handles image file uploads and returns the predicted ASL letter class.

  Args:
    file (UploadFile): The image file uploaded by the user.

  Returns:
    dict: A dictionary containing:
      - "class" (int): The index of the predicted class.
      - "predicted_letter" (str): The predicted ASL letter.
      - "probability" (float): The predicted probability of the predicted class.
      
  Raises:
    JSONResponse: A JSON response with status code 500 and error message if prediction fails.
  """
  image_bytes = await file.read()
  
  try:
    predicted_class, predicted_letter, predicted_prob = predict(image_bytes)
    
    return {"class": predicted_class, "predicted_letter": predicted_letter, "probability": predicted_prob * 100}
  except Exception as e:
    return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})