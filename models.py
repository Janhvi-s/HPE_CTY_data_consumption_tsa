from pydantic import BaseModel

class RequestBody(BaseModel):
    input_data : list