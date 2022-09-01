
# 1. Library imports
import uvicorn
from fastapi import FastAPI
from RFClass import  Orders, Predict

# 2. Create app and model objects
app = FastAPI()
datasetttt = Predict.read_data()
model = Predict.predictions(datasetttt)

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predictions(data : datasetttt):
    #data = datasetttt
    prediction = Predict.predictions(datasetttt)
    return {
        'prediction': prediction, 
        'array': Predict.save_data
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8002)